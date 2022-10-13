"""
Define common methods of models. This done by separating the logic into two parts:
    * Module: This class *only* defines the model architecture and forward pass. This is also done so that others can
      easily copy and adapt the code if necessary.
    * Model: This wrapper class defines all the other logic necessary to use a model in practice: Training, loss
      computation, saving and loading, etc.
"""

# STD
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
import dill
from typing import Dict, Any, Optional, Generator
import os

# EX
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

# PROJECT
import nlp_uncertainty_zoo.utils.metrics as metrics
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun


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
        is_sequence_classifier: bool,
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
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            The device the model is located on.
        build_params: Dict[str, Any]
            Dictionary containing additional parameters used to set up the architecture.
        """
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.is_sequence_classifier = is_sequence_classifier
        self.device = device

        self.single_prediction_uncertainty_metrics = {
            "max_prob": metrics.max_prob,
            "predictive_entropy": metrics.predictive_entropy,
            "dempster_shafer": metrics.dempster_shafer,
            "softmax_gap": metrics.softmax_gap,
        }
        self.multi_prediction_uncertainty_metrics = {}
        self.default_uncertainty_metric = "predictive_entropy"

        super().__init__()

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

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

    @abstractmethod
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
        pass

    def get_uncertainty(
        self,
        input_: torch.LongTensor,
        metric_name: Optional[str] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Get the uncertainty scores for the current batch.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.
        metric_name: Optional[str]
            Name of uncertainty metric being used. If None, use metric defined under the default_uncertainty_metric
            attribute.

        Returns
        -------
        torch.FloatTensor
            Uncertainty scores for the current batch.
        """
        if metric_name is None:
            metric_name = self.default_uncertainty_metric

        logits = self.get_logits(input_, **kwargs)

        with torch.no_grad():
            if metric_name in self.single_prediction_uncertainty_metrics:

                # When model produces multiple predictions, average over them
                if len(logits.shape) == 4:
                    logits = logits.mean(dim=1)

                return self.single_prediction_uncertainty_metrics[metric_name](logits)

            elif metric_name in self.multi_prediction_uncertainty_metrics:
                return self.multi_prediction_uncertainty_metrics[metric_name](logits)

            else:
                raise LookupError(
                    f"Unknown metric '{metric_name}' for class '{self.__class__.__name__}'."
                )

    def get_num_learnable_parameters(self) -> int:
        """
        Return the total number of (learnable) parameters in the model.

        Returns
        -------
        int
            Number of learnable parameters.
        """
        num_parameters = 0

        for param in self.parameters():
            if param.requires_grad:
                flattened_param = torch.flatten(param)
                num_parameters += flattened_param.shape[0]

        return num_parameters


class MultiPredictionMixin:
    """
    Mixin class that is used to bundle certain methods for modules that use multiple predictions to estimate
    uncertainty.
    """

    def __init__(self, num_predictions: int):
        self.num_predictions = num_predictions
        self.multi_prediction_uncertainty_metrics.update(
            {
                "variance": metrics.variance,
                "mutual_information": metrics.mutual_information,
            }
        )


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
        device: Device
            The device the model is located on.
        """
        self.model_name = model_name
        self.module_class = module_class
        self.module = module_class(**model_params, device=device)
        self.model_dir = model_dir
        self.model_params = model_params
        self.device = device
        self.loss_weights = None
        self.to(device)

        # Initialize optimizer and scheduler
        optimizer_class = self.model_params.get("optimizer_class", optim.Adam)
        self.optimizer = optimizer_class(
            self.module.parameters(),
            lr=self.model_params["lr"],
            weight_decay=self.model_params.get("weight_decay", 0),
        )

        self.scheduler = None
        if "scheduler_class" in self.model_params:
            scheduler_class = self.model_params["scheduler_class"]
            self.scheduler = scheduler_class(
                self.optimizer, **self.model_params["scheduler_kwargs"]
            )

        # Check if model directory exists, if not, create
        if model_dir is not None:
            self.full_model_dir = os.path.join(model_dir, model_name)

            if not os.path.exists(self.full_model_dir):
                os.makedirs(self.full_model_dir)

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

            batch_loss = self.get_loss(
                input_ids,
                labels,
                attention_mask=attention_mask,
                wandb_run=wandb_run,
            )

            batch_loss.backward()

            clip_grad_norm_(self.module.parameters(), grad_clip)
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

        return self.module.predict(X, *args, **kwargs)

    def get_uncertainty(
        self,
        input_: torch.LongTensor,
        *args,
        metric_name: Optional[str] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Get the uncertainty scores for the current batch.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.
        metric_name: Optional[str]
            Name of uncertainty metric being used. If None, use metric defined under the default_uncertainty_metric
            attribute.

        Returns
        -------
        torch.FloatTensor
            Uncertainty scores for the current batch.
        """
        return self.module.get_uncertainty(input_, metric_name, *args, **kwargs)

    def eval(self, data_split: DataLoader, wandb_run: Optional[WandBRun] = None) -> torch.Tensor:
        """
        Evaluate a data split.

        Parameters
        ----------
        data_split: DataSplit
            Data split the model should be evaluated on.
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.

        Returns
        -------
        torch.Tensor
            Loss on evaluation split.
        """
        self.module.eval()
        loss = torch.zeros(1)

        for batch in data_split:
            attention_mask, input_ids, labels = (
                batch["attention_mask"].to(self.device),
                batch["input_ids"].to(self.device),
                batch["labels"].to(self.device),
            )
            batch_loss = self.get_loss(
                input_ids,
                labels,
                attention_mask=attention_mask,
                wandb_run=wandb_run,
            )

            loss += batch_loss.detach().cpu()

        self.module.train()

        return loss

    def get_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        wandb_run: Optional[WandBRun] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Get loss for a single batch. This just uses cross-entropy loss, but can be adjusted in subclasses by overwriting
        this function.

        Parameters
        ----------
        X: torch.Tensor
            Batch input.
        y: torch.Tensor
            Batch labels.
        wandb_run: Optional[WandBRun] = None
            Weights and Biases run to track training statistics.

        Returns
        -------
        torch.Tensor
            Batch loss.
        """

        loss_function = nn.CrossEntropyLoss(
            ignore_index=-100, weight=self.loss_weights
        )  # Index that is used for non-masked tokens for MLM
        preds = self.module.forward(X, **kwargs)

        loss = loss_function(
            rearrange(preds, "b t p -> (b t) p"),
            rearrange(y, "b l -> (b l)")
            if not self.module.is_sequence_classifier
            else y,
        )

        return loss

    def _finetune(
        self,
        data_split: DataLoader,
        verbose: bool,
        wandb: Optional[WandBRun] = None,
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
        wandb: Optional[WandBRun]
            Weights & Biases run object to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
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
