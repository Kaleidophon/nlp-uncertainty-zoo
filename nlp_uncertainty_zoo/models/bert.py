"""
Define BERT modules used in this project and make them consistent with the other models in the repository.
"""

# EXT
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# PROJECT
from nlp_uncertainty_zoo.models.model import Module
from nlp_uncertainty_zoo.utils.custom_types import Device


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
        """
        Initialize a BERT module.

        Parameters
        ----------
        bert_name: str
            Name of the underlying BERT, as specified in HuggingFace transformers.
        output_size: int
            Number of classes.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            Device the model should be moved to.
        """

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
            out = torch.tanh(self.bert.pooler.dense(cls_activations))
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
        hidden = torch.tanh(self.bert.pooler.dense(hidden))

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
