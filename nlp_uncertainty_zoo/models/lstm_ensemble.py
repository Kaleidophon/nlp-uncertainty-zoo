"""
Implementation of an ensemble of LSTMs.
"""

# STD
from collections import Counter
from typing import Optional, Dict, Generator, List, Type, Any
import os
from datetime import datetime

# EXT
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
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
        vocab_size: int,
        output_size: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        ensemble_size: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize a LSTM module.

        Parameters
        ----------
        vocab_size: int
            Number of input vocabulary.
        output_size: int
            Number of classes.
        input_size: int
            Dimensionality of input to the first layer (embedding size).
        hidden_size: int
            Size of hidden units.
        num_layers: int
            Number of layers.
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
            vocab_size=vocab_size,
            output_size=output_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            is_sequence_classifier=is_sequence_classifier,
            device=device,
        )
        MultiPredictionMixin.__init__(self, ensemble_size)

        self.ensemble_size = ensemble_size
        self.ensemble_members = nn.ModuleList(
            [
                LSTMModule(
                    vocab_size=vocab_size,
                    output_size=output_size,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    is_sequence_classifier=is_sequence_classifier,
                    device=device,
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

    def get_hidden_representation(
        self, input_: torch.LongTensor, *args, **kwargs
    ) -> torch.FloatTensor:
        members = list(self.ensemble_members._modules.values())

        hidden = torch.stack([
            member.get_hidden_representation(input_, *args, **kwargs) for member in members
        ], dim=0).mean(dim=0)

        return hidden

    @staticmethod
    def get_sequence_representation_from_hidden(hidden: torch.FloatTensor) -> torch.FloatTensor:
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
        vocab_size: int,
        output_size: int,
        input_size: int = 650,
        hidden_size: int = 650,
        num_layers: int = 2,
        dropout: float = 0.2,
        ensemble_size: int = 10,
        init_weight: Optional[float] = 0.6,
        is_sequence_classifier: bool = True,
        lr: float = 0.5,
        weight_decay: float = 0.001,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        scheduler_class: Optional[Type[scheduler._LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None,
        device: Device = "cpu",
        **model_params
    ):
        """
        Initialize a LSTM ensemble.

        Parameters
        ----------
        vocab_size: int
            Number of input vocabulary.
        output_size: int
            Number of classes.
        input_size: int
            Dimensionality of input to the first layer (embedding size). Default is 650.
        hidden_size: int
            Size of hidden units. Default is 650.
        num_layers: int
            Number of layers. Default is 2.
        dropout: float
            Dropout probability. Default is 0.2.
        ensemble_size: int
            Number of members in the ensemble. Default is 10.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step. Default is True.
        lr: float
            Learning rate. Default is 0.5.
        weight_decay: float
            Weight decay term for optimizer. Default is 0.001.
        optimizer_class: Type[optim.Optimizer]
            Optimizer class. Default is Adam.
        scheduler_class: Optional[Type[scheduler._LRScheduler]]
            Learning rate scheduler class. Default is None.
        scheduler_kwargs: Optional[Dict[str, Any]]
            Keyword arguments for learning rate scheduler. Default is None.
        model_dir: Optional[str]
            Directory that model should be saved to.
        device: Device
            Device the model should be moved to.
        """
        super().__init__(
            model_name="lstm_ensemble",
            module_class=LSTMEnsembleModule,
            vocab_size=vocab_size,
            output_size=output_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            ensemble_size=ensemble_size,
            is_sequence_classifier=is_sequence_classifier,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_class=optimizer_class,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            model_dir=model_dir,
            device=device,
        )

        if init_weight is not None:
            for member in self.module.ensemble_members:
                for layer_weights in member.lstm.all_weights:
                    for param in layer_weights:
                        param.data.uniform_(-init_weight, init_weight)

        # Override optimizer and scheduler
        self.optimizer = optimizer_class(
            params=[
                {
                    "params": self.module.ensemble_members[i].parameters(),
                    "lr": lr,
                    "weight_decay": weight_decay
                }
                for i in range(self.module.ensemble_size)
            ]
        )

        if scheduler_class is not None:
            self.scheduler = scheduler_class(
                self.optimizer, **scheduler_kwargs
            )

    def fit(
        self,
        train_split: DataLoader,
        num_training_steps: int,
        valid_split: Optional[DataLoader] = None,
        weight_loss: bool = False,
        grad_clip: float = 10,
        validation_interval: Optional[int] = None,
        early_stopping_pat: int = np.inf,
        early_stopping: bool = False,
        verbose: bool = True,
        wandb_run: Optional[WandBRun] = None,
        **training_kwargs
    ):
        """
        Fit the model to training data. This is a slightly modified function compared to the Model class to accommodate
        ensemble training.

        Parameters
        ----------
        train_split: DataLoader
            Dataset the model is being trained on.
        num_training_steps: int
            Number of training steps until completion.
        valid_split: Optional[DataLoader]
            Validation set the model is being evaluated on if given.
        verbose: bool
            Whether to display information about current loss.
        weight_loss: bool
            Weight classes in loss function. Default is False.
        grad_clip: float
            Parameter grad norm value before it will be clipped. Default is 10.
        validation_interval: Optional[int]
            Interval of training steps between validations on the validation set. If None, the model is evaluated after
            each pass through the training data.
        early_stopping_pat: int
            Patience in number of training steps before early stopping kicks in. Default is np.inf.
        early_stopping: bool
            Whether early stopping should be used. Default is False.
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        if validation_interval is None:
            validation_interval = len(train_split)

        best_val_loss = np.inf
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
