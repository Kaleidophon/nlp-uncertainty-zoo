"""
Define Bert modules used in this project and make them consistent with the other models in the repository.
"""

# EXT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from typing import Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel

# PROJECT
from nlp_uncertainty_zoo.models.model import Module
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun


class BertModule(Module):
    """
    Define a BERT module that implements all the functions implemented in Module.
    """

    def __init__(
        self,
        bert_name: str,
        output_size: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):

        bert = BertModel.from_pretrained(bert_name).to(device)
        hidden_size = bert.config.hidden_size

        super().__init__(
            num_layers=bert.config.num_hidden_layers,
            vocab_size=bert.config.vocab_size,
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            is_sequence_classifier=is_sequence_classifier,
            device=device,
        )

        self.bert = bert
        self.output_size = output_size
        self.sequence_length = bert.config.max_length

        # TODO: Is the custom pooler still necessary?
        # Init custom pooler without tanh activations, copy Bert parameters
        self.custom_bert_pooler = nn.Linear(hidden_size, hidden_size)
        self.custom_bert_pooler.weight = self.bert.pooler.dense.weight
        self.custom_bert_pooler.bias = self.bert.pooler.dense.bias

        self.layer_norm = nn.LayerNorm([hidden_size])
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.
        Returns
        -------
        torch.FloatTensor
            Output predictions for input.
        """
        attention_mask = kwargs["attention_mask"]
        return_dict = self.bert.forward(input_, attention_mask, return_dict=True)

        if self.is_sequence_classifier:
            cls_activations = return_dict["last_hidden_state"][:, 0, :]
            out = torch.tanh(self.custom_bert_pooler(cls_activations))
            out = self.layer_norm(out)
            out = out.unsqueeze(1)

        else:
            activations = return_dict["last_hidden_state"]
            out = self.layer_norm(activations)

        out = self.output(out)

        return out

    def get_logits(
        self, input_: torch.LongTensor, *args, **kwargs
    ) -> torch.FloatTensor:
        """
        Get the logits for an input. Results in a tensor of size batch_size x seq_len x output_size or batch_size x
        num_predictions x seq_len x output_size depending on the model type. Used to create inputs for the uncertainty
        metrics defined in nlp_uncertainty_zoo.metrics.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        return self.forward(input_, *args, **kwargs)

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        """
        Output a probability distribution over classes given an input. Results in a tensor of size batch_size x seq_len
        x output_size or batch_size x num_predictions x seq_len x output_size depending on the model type.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        logits = self.get_logits(input_, *args, **kwargs)
        probabilities = F.softmax(logits, dim=-1)

        return probabilities

    def get_sequence_representation(
        self, hidden: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Define how the representation for an entire sequence is extracted from a number of hidden states. This is
        relevant in sequence classification. For example, this could be the last hidden state for a unidirectional LSTM
        or the first hidden state for a transformer, adding a pooler layer.

        Parameters
        ----------
        hidden: torch.FloatTensor
            Hidden states of a model for a sequence.

        Returns
        -------
        torch.FloatTensor
            Representation for the current sequence.
        """
        hidden = hidden[:, 0, :].unsqueeze(1)
        hidden = torch.tanh(self.custom_bert_pooler(hidden))

        return hidden

    def get_hidden(
        self, input_: torch.LongTensor, *args, **kwargs
    ) -> torch.FloatTensor:
        attention_mask = kwargs["attention_mask"]
        return_dict = self.bert.forward(input_, attention_mask, return_dict=True)

        if self.is_sequence_classifier:
            activations = return_dict["last_hidden_state"][:, 0, :].unsqueeze(1)

        else:
            activations = return_dict["last_hidden_state"]

        return activations


class BertModelMixin:
    """
    Model mixin for BERT models with modified training loop.
    """

    def _epoch_iter(
        self,
        epoch: int,
        data_split: DataLoader,
        progress_bar: Optional[tqdm] = None,
        wandb_run: Optional[WandBRun] = None,
    ) -> torch.Tensor:

        grad_clip = self.model_params.get("grad_clip", np.inf)
        epoch_loss = torch.zeros(1)
        num_batches = len(data_split)

        for i, batch in enumerate(data_split):

            attention_mask, input_ids, labels = (
                batch["attention_mask"].to(self.device),
                batch["input_ids"].to(self.device),
                batch["y"].to(self.device),
            )

            global_batch_num = epoch * len(data_split) + i
            batch_loss = self.get_loss(
                global_batch_num,
                input_ids,
                labels,
                wandb_run,
                attention_mask=attention_mask,
            )

            # Update progress bar and summary writer
            if progress_bar is not None:
                progress_bar.set_description(
                    f"Epoch {epoch + 1}: {i + 1}/{num_batches} | Loss {batch_loss.item():.4f}"
                )
                progress_bar.update(1)

            if wandb_run is not None:
                batch_info = {"Batch train loss": batch_loss}

                if self.scheduler is not None:
                    batch_info["Batch learning rate"] = self.scheduler.get_last_lr()[0]

                wandb_run.log(batch_info, step=global_batch_num)

            epoch_loss += batch_loss.cpu().detach()

            if epoch_loss == np.inf or np.isnan(epoch_loss):
                raise ValueError(f"Loss became NaN or inf during epoch {epoch + 1}.")

            if self.module.training:
                batch_loss.backward()
                clip_grad_norm_(self.module.parameters(), grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # Save memory by setting to None

                if (
                    self.scheduler is not None
                    and self.model_params.get("scheduler_step_or_epoch", "") == "step"
                ):
                    self.scheduler.step()

        return epoch_loss
