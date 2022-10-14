"""
Implementation of an ensemble of LSTMs.
"""

# STD
from collections import Counter
from typing import Optional, Dict, Any, Generator, List
import os
from datetime import datetime

# EXT
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import dill
import numpy as np
from einops import rearrange
import torch
from torch import nn as nn
from tqdm import tqdm

# PROJECT
from nlp_uncertainty_zoo.models.lstm import LSTMModule
from nlp_uncertainty_zoo.models.model import Module, MultiPredictionMixin, Model
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun


class LSTMEnsembleModule(Module, MultiPredictionMixin):
    """
    Implementation for an ensemble of LSTMs.
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        ensemble_size: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize an LSTM.

        Parameters
        ----------
        num_layers: int
            Number of layers.
        vocab_size: int
            Number of input vocabulary.
        input_size: int
            Dimensionality of input to the first layer (embedding size).
        hidden_size: int
            Size of hidden units.
        output_size: int
            Number of classes.
        dropout: float
            Dropout probability.
        ensemble_size: int
            Number of members in the ensemble.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            Device the model should be moved to.
        """
        super().__init__(
            num_layers,
            vocab_size,
            input_size,
            hidden_size,
            output_size,
            is_sequence_classifier,
            device,
        )
        MultiPredictionMixin.__init__(self, ensemble_size)

        self.ensemble_size = ensemble_size
        self.ensemble_members = nn.ModuleList(
            [
                LSTMModule(
                    num_layers,
                    vocab_size,
                    input_size,
                    hidden_size,
                    output_size,
                    dropout,
                    is_sequence_classifier,
                    device,
                )
                for _ in range(ensemble_size)
            ]
        )

    def get_logits(
        self,
        input_: torch.LongTensor,
        *args,
        num_predictions: Optional[int] = None,
        **kwargs,
    ):

        if num_predictions is None:
            q, r = 1, 0
        else:
            q, r = divmod(num_predictions, len(self.ensemble_members))

        members = list(self.ensemble_members._modules.values())

        out = torch.stack(
            [member.get_logits(input_) for member in q * members + members[:r]], dim=1
        )

        return out

    def forward(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        preds = self.get_logits(input_)

        return preds

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        preds = Module.predict(self, input_, *args, **kwargs)
        preds = preds.mean(dim=1)

        return preds

    def to(self, device: Device):
        """
        Move model to another device.

        Parameters
        ----------
        device: Device
            Device the model should be moved to.
        """
        for member in self.ensemble_members:
            member.to(device)

    @staticmethod
    def get_sequence_representation(hidden: torch.FloatTensor) -> torch.FloatTensor:
        """
        Create a sequence representation from an ensemble of LSTMs.

        Parameters
        ----------
        hidden: torch.FloatTensor
            Hidden states of a model for a sequence.

        Returns
        -------
        torch.FloatTensor
            Representation for the current sequence.
        """
        return hidden[:, -1, :].unsqueeze(1)


class LSTMEnsemble(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "lstm_ensemble",
            LSTMEnsembleModule,
            model_params,
            model_dir,
            device,
        )

        if "init_weight" in model_params:
            init_weight = model_params["init_weight"]

            for member in self.module.ensemble_members:
                for layer_weights in member.lstm.all_weights:
                    for param in layer_weights:
                        param.data.uniform_(-init_weight, init_weight)

        # Override optimizer and scheduler
        optimizer_class = self.model_params.get("optimizer_class", optim.Adam)
        self.optimizer = optimizer_class(
            params=[
                {
                    "params": self.module.ensemble_members[i].parameters(),
                    "lr": self.model_params["lr"],
                    "weight_decay": self.model_params.get("weight_decay", 0)
                }
                for i in range(self.module.ensemble_size)
            ]
        )

        self.scheduler = None
        if "scheduler_class" in self.model_params:
            scheduler_class = self.model_params["scheduler_class"]
            self.scheduler = scheduler_class(
                self.optimizer, **self.model_params["scheduler_kwargs"]
            )

    def fit(
        self,
        train_split: DataLoader,
        valid_split: Optional[DataLoader] = None,
        verbose: bool = True,
        weight_loss: bool = False,
        wandb_run: Optional[WandBRun] = None,
    ):
        """
        Fit the model to training data.

        Parameters
        ----------
        train_split: DataLoader
            Dataset the model is being trained on.
        valid_split: Optional[DataLoader]
            Validation set the model is being evaluated on if given.
        verbose: bool
            Whether to display information about current loss.
        weight_loss: bool
            Weight classes in loss function. Default is False.
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        num_training_steps = self.model_params["num_training_steps"]
        best_val_loss = np.inf
        grad_clip = self.model_params.get("grad_clip", np.inf)
        validation_interval = self.model_params["validation_interval"]
        early_stopping_pat = self.model_params.get("early_stopping_pat", np.inf)
        early_stopping = self.model_params.get("early_stopping", True)
        num_no_improvements = 0
        progress_bar = tqdm(total=num_training_steps) if verbose else None
        best_model = dict(self.__dict__)

        # Compute loss weights
        if weight_loss:
            counter = Counter()

            for batch in train_split:
                labels = batch["labels"]

                # Flatten label lists with sum(), then filter ignore label
                counter.update(filter(lambda label: label != -100, sum(labels.tolist(), [])))

            self.loss_weights = torch.zeros(self.module.output_size, device=self.device)

            for key, freq in counter.items():
                self.loss_weights[key] = freq

            del counter

            self.loss_weights /= torch.sum(self.loss_weights)
            self.loss_weights = 1 - self.loss_weights

        def batch_generator(train_split: DataLoader) -> Generator[Dict[str, torch.Tensor], None, None]:
            """
            Quick generator that outputs batches indefinitely.
            """
            while True:
                for batch in train_split:
                    yield batch

        for training_step, batch in enumerate(batch_generator(train_split)):

            if training_step == num_training_steps:
                break

            self.module.train()

            attention_mask, input_ids, labels = (
                batch["attention_mask"].to(self.device),
                batch["input_ids"].to(self.device),
                batch["labels"].to(self.device),
            )

            ### Only change for ensemble class: Backprop through ensemble member losses separately

            batch_losses = self.get_train_loss(
                input_ids,
                labels,
                attention_mask=attention_mask,
                wandb_run=wandb_run,
            )

            for batch_loss in batch_losses:
                batch_loss.backward(retain_graph=True)

            batch_loss = torch.stack(batch_losses).mean().detach()  # Still get mean for tracking purposes

            # Clip the parameter gradients member-wise
            for member in self.module.ensemble_members:
                clip_grad_norm_(member.parameters(), grad_clip)

            ### Change end

            self.optimizer.step()
            self.optimizer.zero_grad(
                set_to_none=True
            )  # Save memory by setting to None

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

                if wandb_run is not None:
                    wandb_run.log({"batch_learning_rate": self.scheduler.get_last_lr()[0]})

            batch_loss = batch_loss.cpu().detach().item()
            if batch_loss == np.inf or np.isnan(batch_loss):
                raise ValueError(f"Loss became NaN or inf during step {training_step + 1}.")

            # Update progress bar and summary writer
            if verbose:
                progress_bar.set_description(
                    f"Step {training_step + 1}: Train Loss {batch_loss:.4f}"
                )
                progress_bar.update(1)

            if wandb_run is not None:
                wandb_run.log({"batch_train_loss": batch_loss})

            # Get validation loss
            if valid_split is not None and training_step % validation_interval == 0 and training_step > 0:
                self.module.eval()

                with torch.no_grad():
                    val_loss = self.eval(valid_split, wandb_run=wandb_run)

                if wandb_run is not None:
                    import wandb
                    wandb.log({"val_loss": val_loss.item()})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss.item()

                    if early_stopping:
                        best_model = dict(self.__dict__)

                else:
                    num_no_improvements += 1

                    if num_no_improvements > early_stopping_pat:
                        break

        # Set current model to best model found, otherwise use last
        if early_stopping:
            self.__dict__ = best_model
            del best_model

        # Additional training step, e.g. temperature scaling on val
        if valid_split is not None:
            self._finetune(valid_split, verbose, wandb_run)

        # Save model if applicable
        if self.model_dir is not None:
            timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))
            torch.save(
                self,
                os.path.join(
                    self.full_model_dir,
                    f"{best_val_loss:.2f}_{timestamp}.pt",
                ),
                pickle_module=dill,
                _use_new_zipfile_serialization=False,
            )

        result_dict = {
            "model_name": self.model_name,
            "train_loss": batch_loss,
            "best_val_loss": best_val_loss,
        }

        return result_dict

    def get_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        wandb_run: Optional[WandBRun] = None,
        **kwargs,
    ) -> torch.Tensor:
        loss_function = nn.CrossEntropyLoss(
            ignore_index=-100, weight=self.loss_weights
        )  # Index that is used for non-masked tokens for MLM
        total_loss = torch.zeros(1, device=self.device)
        preds = self.module.forward(X, **kwargs)

        for n in range(self.module.ensemble_size):

            total_loss += loss_function(
                rearrange(preds[:, n], "b t p -> (b t) p"),
                rearrange(y, "b l -> (b l)")
                if not self.module.is_sequence_classifier
                else y,
            )

        total_loss /= self.module.ensemble_size

        return total_loss

    def get_train_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        wandb_run: Optional[WandBRun] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        loss_function = nn.CrossEntropyLoss(
            ignore_index=-100, weight=self.loss_weights
        )  # Index that is used for non-masked tokens for MLM
        total_loss = []
        preds = self.module.forward(X, **kwargs)

        for n in range(self.module.ensemble_size):

            loss = loss_function(
                rearrange(preds[:, n], "b t p -> (b t) p"),
                rearrange(y, "b l -> (b l)")
                if not self.module.is_sequence_classifier
                else y,
            )
            total_loss.append(loss)

        return total_loss
