"""
Define common methods of a module class.
"""

# STD
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os

# EXT
from codecarbon import OfflineEmissionsTracker
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# PROJECT
from src.datasets import DataSplit, TextDataset
from src.types import Device


class Model(ABC, nn.Module):
    """
    Abstract model class, defining how the forward pass of a model looks and how the architecture is being built.
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


class Module(ABC):
    """
    Abstract module class. It is a wrapper that defines data loading, batching, training and evaluation loops, so that
    the core model class can only define the model's forward pass.
    """

    def __init__(
        self,
        model_name: str,
        model_class: type,
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
        model_class: type
            Class of the model that is being wrapped.
        model_params: Dict[str, Any]
            Parameters to initialize the model.
        train_params: Dict[str, Any]
            Parameters for model training.
        device: Device
            The device the model is located on.
        """
        self.model_name = model_name
        self.model_class = model_class
        self.model = model_class(**model_params, device=device)
        self.train_params = train_params
        self.model_dir = model_dir
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_params["lr"])
        self.to(device)

        # Check if model directory exists, if not, create
        if model_dir is not None:
            self.full_model_dir = os.path.join(model_dir, model_name)

            if not os.path.exists(self.full_model_dir):
                os.mkdir(self.full_model_dir)

    def fit(
        self,
        dataset: TextDataset,
        verbose: bool = True,
        summary_writer: Optional[SummaryWriter] = None,
    ):
        """
        Fit the model to training data.

        Parameters
        ----------
        dataset: TextDataset
            Dataset the model is being trained on.
        verbose: bool
            Whether to display information about current loss.
        summary_writer: Optional[SummaryWriter]
            Summary writer to track training statistics. Training and validation loss (if applicable) are tracked by
            default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        device = self.device
        num_epochs = self.train_params["num_epochs"]

        best_val_loss = np.inf
        early_stopping_pat = self.train_params.get("early_stopping_pat", np.inf)
        num_no_improvements = 0
        # TODO: Set country code in config
        tracker = OfflineEmissionsTracker(
            project_name="nlp-uncertainty-zoo",
            country_iso_code="DNK",
            output_dir=self.model_dir,
        )
        tracker.start()
        total_steps = num_epochs * len(dataset.train) + num_epochs * len(dataset.valid)

        with tqdm(total=total_steps) as progress_bar:
            for epoch in range(self.train_params["num_epochs"]):
                self.model.train()
                train_loss = self._epoch_iter(
                    epoch,
                    dataset.train.to(device),
                    progress_bar if verbose else None,
                    summary_writer,
                )

                # Update progress bar and summary writer
                if verbose is not None:
                    progress_bar.set_description(
                        f"Epoch {epoch + 1} / {num_epochs}: Train Loss {train_loss.item():.4f}"
                    )
                    progress_bar.update(1)

                if summary_writer is not None:
                    summary_writer.add_scalar(
                        "Epoch train loss", train_loss.item(), epoch
                    )

                # Get validation loss
                self.model.eval()
                val_loss = self._epoch_iter(
                    epoch, dataset.valid.to(device), progress_bar if verbose else None
                )

                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch val loss", val_loss, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = best_val_loss

                    if self.model_dir is not None:
                        torch.save(
                            self,
                            os.path.join(
                                self.full_model_dir,
                                f"{epoch + 1}_{val_loss.item():.2f}.pt",
                            ),
                        )

                else:
                    num_no_improvements += 1

                    if num_no_improvements > early_stopping_pat:
                        break

        # Additional training step, e.g. temperature scaling on val
        self._finetune(dataset.valid.to(device))

        # Stop emission tracking
        tracker.stop()

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
        X.to(self.device)

        return self.model(X)

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
        epoch_loss = torch.zeros(1)
        num_batches = len(data_split)

        for i, (X, y) in enumerate(data_split):
            batch_loss = self.get_loss(i, X, y, summary_writer)

            # Update progress bar and summary writer
            if progress_bar is not None:
                progress_bar.set_description(
                    f"Epoch {epoch + 1}: {i+1}/{num_batches} | Loss {batch_loss.item():.4f}"
                )
                progress_bar.update(1)

            if summary_writer is not None:
                summary_writer.add_scalar(
                    "Batch train loss", batch_loss, epoch * len(data_split) + i
                )

            epoch_loss += batch_loss

            if self.model.training:
                batch_loss.backward()
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

        loss_function = nn.NLLLoss()
        preds = self.model(X)
        batch_size, sequence_length, output_size = preds.shape
        preds = preds.reshape(batch_size * sequence_length, output_size)
        y = y.reshape(batch_size * sequence_length)

        loss = loss_function(preds, y)

        return loss

    def _finetune(self, data_loader: data.DataLoader) -> torch.Tensor:
        """
        Do an additional training / fine-tuning step, which is required for some models. Is being overwritten in some
        subclasses.

        Parameters
        ----------
        data_loader: data.DataLoader
            Data loader for fine-tuning data.
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
        self.model.to(device)

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
        Module
            Loaded model.
        """
        with open(model_path, "rb") as pickled_model:
            return torch.load(pickled_model)
