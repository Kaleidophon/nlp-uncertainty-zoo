"""
Define common methods of models. This done by separating the logic into two parts:
    * Module: This class *only* defines the model architecture and forward pass. This is also done so that others can
      easily copy and adapt the code if necessary.
    * Model: This wrapper class defines all the other logic necessary to use a model in practice: Training, loss
      computation, saving and loading, etc.
"""

# STD
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, Optional
import os

# EXT
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# PROJECT
from src.datasets import DataSplit, TextDataset
from src.evaluation import evaluate
from src.types import Device


class Module(ABC, nn.Module):
    """
    Abstract module class, defining how the forward pass of a model looks.
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: Device,
        **build_params,
    ):
        """
        Initialize a model.

        Parameters
        ----------
        num_layers: int
            Number of model layers.
        vocab_size: int
            Vocabulary size.
        input_size: int
            Dimensionality of input to model.
        hidden_size: int
            Size of hidden representations.
        output_size: int
            Size of output of model.
        build_params: Dict[str, Any]
            Dictionary containing additional parameters used to set up the architecture.
        """
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        super().__init__()

    @abstractmethod
    def forward(self, input_: torch.LongTensor) -> torch.FloatTensor:
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
        pass


class Model(ABC):
    """
    Abstract model class. It is a wrapper that defines data loading, batching, training and evaluation loops, so that
    the core module class can only define the model's forward pass.
    """

    def __init__(
        self,
        model_name: str,
        module_class: type,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        """
        Initialize a module.

        Parameters
        ----------
        model_name: str
            Name of the model.
        module_class: type
            Class of the model that is being wrapped.
        model_params: Dict[str, Any]
            Parameters to initialize the model.
        train_params: Dict[str, Any]
            Parameters for model training.
        device: Device
            The device the model is located on.
        """
        self.model_name = model_name
        self.module_class = module_class
        self.module = module_class(**model_params, device=device)
        self.train_params = train_params
        self.model_dir = model_dir
        self.device = device
        self.to(device)

        # Initialize optimizer and scheduler
        # TODO: Make optimizer an option
        self.optimizer = optim.SGD(
            self.module.parameters(),
            lr=self.train_params["lr"],
            weight_decay=self.train_params["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.train_params["milestones"],
            gamma=self.train_params["gamma"],
        )

        # Check if model directory exists, if not, create
        if model_dir is not None:
            self.full_model_dir = os.path.join(model_dir, model_name)

            if not os.path.exists(self.full_model_dir):
                os.mkdir(self.full_model_dir)

    def fit(
        self,
        dataset: TextDataset,
        validate: bool = True,
        verbose: bool = True,
        summary_writer: Optional[SummaryWriter] = None,
    ):
        """
        Fit the model to training data.

        Parameters
        ----------
        dataset: TextDataset
            Dataset the model is being trained on.
        validate: bool
            Indicate whether model should also be evaluated on the validation set.
        verbose: bool
            Whether to display information about current loss.
        summary_writer: Optional[SummaryWriter]
            Summary writer to track training statistics. Training and validation loss (if applicable) are tracked by
            default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        num_epochs = self.train_params["num_epochs"]
        best_val_loss = np.inf
        early_stopping_pat = self.train_params.get("early_stopping_pat", np.inf)
        early_stopping = self.train_params.get("early_stopping", True)
        num_no_improvements = 0
        total_steps = num_epochs * len(dataset.train)
        progress_bar = tqdm(total=total_steps) if verbose else None
        best_model = deepcopy(self)

        for epoch in range(self.train_params["num_epochs"]):
            self.module.train()

            train_loss = self._epoch_iter(
                epoch,
                dataset.train,
                progress_bar,
                summary_writer,
            )

            # Update progress bar and summary writer
            if verbose:
                progress_bar.set_description(
                    f"Epoch {epoch + 1} / {num_epochs}: Train Loss {train_loss.item():.4f}"
                )
                progress_bar.update(1)

            if summary_writer is not None:
                summary_writer.add_scalar("Epoch train loss", train_loss.item(), epoch)

            # Get validation loss
            if validate:
                self.module.eval()

                with torch.no_grad():
                    val_loss = evaluate(self, dataset, dataset.valid)

                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch val score", val_loss, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss.item()

                    if early_stopping:
                        best_model = deepcopy(self)

                else:
                    num_no_improvements += 1

                    if num_no_improvements > early_stopping_pat:
                        break

            # Update scheduler
            self.scheduler.step(epoch=epoch)

        # Set current model to best model found, otherwise use last
        if early_stopping:
            self.__dict__.update(best_model.__dict__)
            del best_model

        # Additional training step, e.g. temperature scaling on val
        if validate:
            self._finetune(dataset.valid, verbose, summary_writer)

        # Save model if applicable
        if self.model_dir is not None:
            timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))
            torch.save(
                self,
                os.path.join(
                    self.full_model_dir,
                    f"{best_val_loss:.2f}_{timestamp}.pt",
                ),
                _use_new_zipfile_serialization=False,
            )

        # Make a nice result dict for knockknock
        result_dict = {
            "model_name": self.model_name,
            "train_loss": train_loss.item(),
            "best_val_loss": best_val_loss,
        }

        return result_dict

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Make a prediction for some input.

        Parameters
        ----------
        X: torch.Tensor
            Input data points.

        Returns
        -------
        torch.Tensor
            Predictions.
        """
        X = X.to(self.device)

        return self.module(X)

    def eval(self, data_split: DataSplit) -> torch.Tensor:
        """
        Evaluate a data split.

        Parameters
        ----------
        data_split: DataSplit
            Data split the model should be evaluated on.

        Returns
        -------
        torch.Tensor
            Loss on evaluation split.
        """
        self.module.eval()
        loss = self._epoch_iter(0, data_split)
        self.module.train()

        return loss

    def _epoch_iter(
        self,
        epoch: int,
        data_split: DataSplit,
        progress_bar: Optional[tqdm] = None,
        summary_writer: Optional[SummaryWriter] = None,
    ) -> torch.Tensor:
        """
        Perform one training epoch.

        Parameters
        ----------
        epoch: int
            Number of the current epoch.
        data_split: DataSplit
            Current data split.
        progress_bar: Optional[tqdm]
            Progress bar used to display information about current run.
        summary_writer: SummaryWriter
            Summary writer to track training statistics.
        """
        grad_clip = self.train_params.get("grad_clip", np.inf)
        epoch_loss = torch.zeros(1)
        num_batches = len(data_split)

        for i, (X, y) in enumerate(data_split):
            X, y = X.to(self.device), y.to(self.device)
            global_batch_num = epoch * len(data_split) + i
            batch_loss = self.get_loss(global_batch_num, X, y, summary_writer)

            # Update progress bar and summary writer
            if progress_bar is not None:
                progress_bar.set_description(
                    f"Epoch {epoch + 1}: {i+1}/{num_batches} | Loss {batch_loss.item():.4f}"
                )
                progress_bar.update(1)

            if summary_writer is not None:
                summary_writer.add_scalar(
                    "Batch train loss", batch_loss, global_batch_num
                )
                summary_writer.add_scalar(
                    "Batch learning rate",
                    self.scheduler.get_last_lr()[0],
                    global_batch_num,
                )

            epoch_loss += batch_loss.cpu().detach()

            if epoch_loss == np.inf or np.isnan(epoch_loss):
                raise ValueError(f"Loss became NaN or inf during epoch {epoch + 1}.")

            if self.module.training:
                batch_loss.backward()

                clip_grad_norm_(self.module.parameters(), grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

        return epoch_loss

    def get_loss(
        self,
        n_batch: int,
        X: torch.Tensor,
        y: torch.Tensor,
        summary_writer: Optional[SummaryWriter] = None,
    ) -> torch.Tensor:
        """
        Get loss for a single batch. This just uses cross-entropy loss, but can be adjusted in subclasses by overwriting
        this function.

        Parameters
        ----------
        n_batch: int
            Number of the current batch.
        X: torch.Tensor
            Batch input.
        y: torch.Tensor
            Batch labels.
        summary_writer: SummaryWriter
            Summary writer to track training statistics.

        Returns
        -------
        torch.Tensor
            Batch loss.
        """

        loss_function = nn.CrossEntropyLoss()
        preds = self.module(X)
        batch_size, sequence_length, output_size = preds.shape
        preds = preds.reshape(batch_size * sequence_length, output_size)
        y = y.reshape(batch_size * sequence_length)

        loss = loss_function(preds, y)

        return loss

    def _finetune(
        self,
        data_split: DataSplit,
        verbose: bool,
        summary_writer: Optional[SummaryWriter] = None,
    ):
        """
        Do an additional training / fine-tuning step, which is required for some models. Is being overwritten in some
        subclasses.

        Parameters
        ----------
        data_split: data.DataSplit
            Data split for fine-tuning step.
        verbose: bool
            Whether to display information about current loss.
        summary_writer: Optional[SummaryWriter]
            Summary writer to track training statistics. Training and validation loss (if applicable) are tracked by
            default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        pass

    def to(self, device: Device):
        """
        Move model to another device.

        Parameters
        ----------
        device: Device
            Device the model should be moved to.
        """
        self.module.to(device)

    @staticmethod
    def load(model_path: str):
        """
        Load model from path.

        Parameters
        ----------
        model_path: str
            Path model was saved to.

        Returns
        -------
        Model
            Loaded model.
        """
        return torch.load(model_path)
