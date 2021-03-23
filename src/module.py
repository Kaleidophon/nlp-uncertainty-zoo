"""
Define common methods of a module class.
"""

# STD
from abc import ABC
from typing import Dict, Any, Optional, Union

# EXT
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Custom types
Device = Union[torch.device, str]


# TODO: Add tracking of CO2 emissions
# TODO: Model superclass with build_architecture()


class Module(ABC):
    """
    Abstract module class. It is a wrapper that defines data loading, batching, training and evaluation loops, so that
    the core model class can only define the model's forward pass.
    """

    def __init__(
        self,
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
        model_class: type
            Class of the model that is being wrapped.
        model_params: Dict[str, Any]
            Parameters to initialize the model.
        train_params: Dict[str, Any]
            Parameters for model training.
        device: Device
            The device the model is located on.
        """
        self.model_class = model_class
        self.model = model_class(**model_params)
        self.train_params = train_params
        self.model_dir = model_dir
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_params["lr"])
        self.to(device)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor],
        y_val: Optional[torch.Tensor],
        verbose: bool = True,
        summary_writer: Optional[SummaryWriter] = None,
    ):
        """
        Fit the model to training data.

        Parameters
        ----------
        X_train: torch.Tensor
            Training data.
        y_train: torch.Tensor
            Training labels.
        X_val: Optional[torch.Tensor]
            Validation data.
        y_val: Optional[torch.Tensor]
            Validation labels.
        verbose: bool
            Whether to display information about current loss.
        summary_writer: Optional[SummaryWriter]
            Summary writer to track training statistics. Training and validation loss (if applicable) are tracked by
            default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        if None not in (X_val, y_val):
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)
            dl_val = self._init_data_loader(X_val, y_val)

        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        dl_train = self._init_data_loader(X_train, y_train)
        num_epochs = self.train_params["num_epochs"]

        best_val_loss = np.inf
        early_stopping_pat = self.train_params.get("early_stopping_pat", np.inf)
        num_no_improvements = 0

        with tqdm(total=num_epochs) as progress_bar:
            for epoch in tqdm(self.train_params["num_epochs"]):
                self.model.train()
                train_loss = self._epoch_iter(epoch, dl_train, summary_writer)

                # Update progress bar and summary writer
                if verbose is not None:
                    progress_bar.set_description(
                        f"Epoch {epoch + 1} / {num_epochs}: Train Loss {train_loss:.4f}"
                    )
                    progress_bar.update(1)

                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch train loss", train_loss, epoch)

                if None not in (X_val, y_val):
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    self.model.eval()
                    val_loss = self._epoch_iter(epoch, dl_val, summary_writer)

                    if summary_writer is not None:
                        summary_writer.add_scalar("Epoch val loss", val_loss, epoch)

                    if val_loss < best_val_loss:
                        best_val_loss = best_val_loss

                        with open(
                            f"{self.model_dir}/{epoch}_{val_loss:.2f}.pt"
                        ) as model_path:
                            torch.save(self, model_path)

                    else:
                        num_no_improvements += 1

                        if num_no_improvements > early_stopping_pat:
                            break

        # Additional training step, e.g. temperature scaling on val
        if None not in (X_val, y_val):
            dl_val = self._init_data_loader(X_val, y_val)
            self._finetune(dl_val)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
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

    def _init_data_loader(self, X: torch.Tensor, y: torch.Tensor) -> data.DataLoader:
        ...  # TODO

    def _epoch_iter(
        self,
        n_epoch: int,
        data_loader: data.DataLoader,
        summary_writer: Optional[SummaryWriter] = None,
    ) -> torch.Tensor:
        """
        Perform one training epoch.

        Parameters
        ----------
        n_epoch: int
            Number of the current epoch.
        data_loader: data.DataLoader
            Data loader for current data split.
        summary_writer: SummaryWriter
            Summary writer to track training statistics.
        """
        epoch_loss = torch.zeros(1)

        for i, (X, y) in enumerate(data_loader):
            batch_loss = self.get_loss(i, X, y, summary_writer)

            if summary_writer is not None:
                summary_writer.add_scalar(
                    "Batch train loss", batch_loss, n_epoch * len(data_loader) + i
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
