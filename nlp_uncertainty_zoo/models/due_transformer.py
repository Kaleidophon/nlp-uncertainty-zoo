"""
Implementation of Deterministic Uncertainty Estimation (DUE) Transformer by
`Van Amersfoort et al., 2021 <https://arxiv.org/pdf/2102.11409.pdf>`_.
"""

# STD
import math
from typing import Dict, Any, Optional, List

# EXT
import torch
from einops import rearrange
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# PROJECT
from nlp_uncertainty_zoo.models.due_utils import _get_initial_inducing_points, _get_initial_lengthscale, GP
from nlp_uncertainty_zoo.models.spectral import (
    SpectralTransformerModule,
    SpectralBertModule,
)
from nlp_uncertainty_zoo.models.model import MultiPredictionMixin, Model
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun


class DUEMixin:
    """
    Implementation of Deterministic Uncertainty Estimation (DUE) by
    `Van Amersfoort et al., 2021 <https://arxiv.org/pdf/2102.11409.pdf>`_ as a Mixin class. This is done to avoid
    code redundancies.
    """

    def __init__(
        self, num_inducing_samples: int, num_inducing_points: int, kernel_type: str
    ):
        self.num_inducing_samples = num_inducing_samples
        self.num_inducing_points = num_inducing_points
        self.kernel_type = kernel_type

        self.gp = None
        self.likelihood = None
        self.loss_function = None

    def init_gp(self, train_data: DataLoader, ignore_indices: List[int] = [-100, 0, 1, 2, 3, 4]):
        """
        Initialize the Gaussian Process layer together with the likelihood and loss function.

        Parameters
        ----------
        train_data: DataSplit
            Training split.
        num_instances: int
            Number of instances being sampled to initialize the GP inducing points and length scale.
        """
        # Compute how many batches need to be sampled to initialize the inducing points when using batches of
        # batch_size and length sequence_length
        batch_size = train_data.dataset[0]["input_ids"].shape[0]

        # Extract feature representations for sampled batches
        batch_representations = []
        batch_masks = []
        num_representations = 0

        with torch.no_grad():
            train_data.shuffle = True
            while num_representations <= self.num_inducing_samples:
                batch = next(iter(train_data))
                input_ = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                batch_representations.append(
                    self.get_hidden(input_, attention_mask=attention_mask)
                )
                batch_mask = rearrange(torch.all(torch.stack([input_ != idx for idx in ignore_indices]), dim=0), "b s -> (b s)")
                batch_masks.append(batch_mask)
                num_representations += batch_mask.int().sum()

        representations = torch.cat(batch_representations, dim=0)
        mask = torch.cat(batch_masks, dim=0)
        representations = rearrange(representations, "b s h -> (b s) h")
        representations = representations[mask]

        initial_inducing_points = _get_initial_inducing_points(
            representations.cpu().numpy(), self.num_inducing_points
        )
        initial_length_scale = _get_initial_lengthscale(representations)

        self.gp = GP(
            num_outputs=self.output_size,
            initial_lengthscale=initial_length_scale,
            initial_inducing_points=initial_inducing_points,
            kernel=self.kernel_type,
        ).to(self.device)
        self.likelihood = SoftmaxLikelihood(
            num_features=self.input_size,
            num_classes=self.output_size,
            mixing_weights=False,
        ).to(self.device)
        self.loss_function = VariationalELBO(
            self.likelihood, self.gp, num_data=len(train_data),
        ).to(self.device)

    def forward(self, input_: torch.LongTensor, ignore_mask: Optional[torch.BoolTensor] = None, **kwargs):
        out = self.get_hidden(input_, **kwargs)

        if self.is_sequence_classifier:
            out = self.get_sequence_representation(out)

        out = rearrange(out, "b s h -> (b s) h").float()

        if ignore_mask is not None:
            out = out[ignore_mask]

        mvn = self.gp(out)

        return mvn

    def predict(self, input_: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size = input_.shape[0]

        with torch.no_grad():
            out = DUEMixin.forward(self, input_, **kwargs)
            out = self.likelihood(out)
            out = out.logits.mean(dim=0)

        out = rearrange(out, "(b t) p -> b t p", b=batch_size)
        preds = F.softmax(out, dim=-1)

        return preds

    def get_logits(
        self,
        input_: torch.LongTensor,
        *args,
        num_predictions: Optional[int] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Get the logits for an input. Results in a tensor of size batch_size x seq_len x output_size or batch_size x
        num_predictions x seq_len x output_size depending on the model type. Used to create inputs for the uncertainty
        metrics defined in nlp_uncertainty_zoo.metrics.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.
        num_predictions: Optional[int]
            Number of samples used to make predictions.

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        if not num_predictions:
            num_predictions = self.num_predictions

        batch_size = input_.shape[0]

        mvn = DUEMixin.forward(self, input_, **kwargs)
        predictions = mvn.sample(sample_shape=torch.Size((num_predictions,)))
        predictions = rearrange(
            predictions, "n (b s) o  -> b n s o", b=batch_size, n=num_predictions
        )

        return predictions


class DUETransformerModule(SpectralTransformerModule, MultiPredictionMixin, DUEMixin):
    """
    Implementation of Deterministic Uncertainty Estimation (DUE) Transformer by
    `Van Amersfoort et al., 2021 <https://arxiv.org/pdf/2102.11409.pdf>`_.
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        input_dropout: float,
        dropout: float,
        num_heads: int,
        sequence_length: int,
        num_predictions: int,
        num_inducing_samples: int,
        num_inducing_points: int,
        spectral_norm_upper_bound: float,
        kernel_type: str,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):

        """
        Initialize a DDU transformer.

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
        input_dropout: float
            Input dropout added to embeddings.
        dropout: float
            Dropout rate.
        num_heads: int
            Number of self-attention heads per layer.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings.
        spectral_norm_upper_bound: float
            Set a limit when weight matrices will be spectrally normalized if their eigenvalue surpasses it.
        kernel_type: str
            Define the type of kernel used. Can be one of {'RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ'}.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            Device the model is located on.
        """
        super().__init__(
            num_layers,
            vocab_size,
            input_size,
            hidden_size,
            output_size,
            input_dropout,
            dropout,
            num_heads,
            sequence_length,
            spectral_norm_upper_bound,
            is_sequence_classifier,
            device,
        )
        MultiPredictionMixin.__init__(self, num_predictions)
        DUEMixin.__init__(self, num_inducing_samples, num_inducing_points, kernel_type)

        self.layer_norm = nn.LayerNorm([input_size])

    def get_logits(
        self, input_: torch.LongTensor, *args, **kwargs
    ) -> torch.FloatTensor:
        return DUEMixin.get_logits(self, input_, *args, **kwargs)

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        return DUEMixin.predict(self, input_, *args, **kwargs)


class DUETransformer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "ddu_transformer",
            DUETransformerModule,
            model_params,
            model_dir,
            device,
        )

    def fit(
        self,
        train_split: DataLoader,
        valid_split: Optional[DataLoader] = None,
        verbose: bool = True,
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
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        # Retrieve inducing points and length scale from training set to initialize the GP
        self.module.init_gp(train_split)

        return super().fit(train_split, valid_split, verbose, wandb_run)

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.module.predict(X, *args, **kwargs)

    def get_loss(
        self,
        n_batch: int,
        X: torch.Tensor,
        y: torch.Tensor,
        wandb_run: Optional[WandBRun] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Get loss for a single batch. This uses the Variational ELBO instead of a cross-entropy loss.

        Parameters
        ----------
        n_batch: int
            Number of the current batch.
        X: torch.Tensor
            Batch input.
        y: torch.Tensor
            Batch labels.
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.

        Returns
        -------
        torch.Tensor
            Batch loss.
        """
        if not self.module.is_sequence_classifier:
            y = rearrange(y, "b t -> (b t)")

        preds = DUEMixin.forward(self.module, X, **kwargs)
        loss = -self.module.loss_function(preds, y)

        return loss


class DUEBertModule(SpectralBertModule, MultiPredictionMixin, DUEMixin):
    """
    Implementation of Deterministic Uncertainty Estimation (DUE) Transformer by
    `Van Amersfoort et al., 2021 <https://arxiv.org/pdf/2102.11409.pdf>`_ in the form of a pre-trained Bert.
    """

    def __init__(
        self,
        bert_name: str,
        output_size: int,
        spectral_norm_upper_bound: float,
        num_predictions: int,
        num_inducing_samples: int,
        num_inducing_points: int,
        kernel_type: str,
        is_sequence_classifier: bool,
        device: Device,
        ignore_label: int = -100,
        **build_params,
    ):
        super().__init__(
            bert_name,
            output_size,
            spectral_norm_upper_bound,
            is_sequence_classifier,
            device,
            **build_params,
        )
        self.ignore_label = ignore_label
        MultiPredictionMixin.__init__(self, num_predictions)
        DUEMixin.__init__(self, num_inducing_samples, num_inducing_points, kernel_type)

    def get_logits(
        self, input_: torch.LongTensor, *args, **kwargs
    ) -> torch.FloatTensor:
        return DUEMixin.get_logits(self, input_, *args, **kwargs)

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        return DUEMixin.predict(self, input_, *args, **kwargs)


class DUEBert(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        bert_name = model_params["bert_name"]
        super().__init__(
            f"due-{bert_name}",
            DUEBertModule,
            model_params,
            model_dir,
            device,
        )

    def fit(
        self,
        train_split: DataLoader,
        valid_split: Optional[DataLoader] = None,
        verbose: bool = True,
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
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        # Retrieve inducing points and length scale from training set to initialize the GP
        self.module.init_gp(train_split)

        return super().fit(train_split, valid_split, verbose, wandb_run)

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.module.predict(X, *args, **kwargs)

    def get_loss(
        self,
        n_batch: int,
        X: torch.Tensor,
        y: torch.Tensor,
        wandb_run: Optional[WandBRun] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Get loss for a single batch. This uses the Variational ELBO instead of a cross-entropy loss.

        Parameters
        ----------
        n_batch: int
            Number of the current batch.
        X: torch.Tensor
            Batch input.
        y: torch.Tensor
            Batch labels.
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.

        Returns
        -------
        torch.Tensor
            Batch loss.
        """
        if not self.module.is_sequence_classifier:
            y = rearrange(y, "b t -> (b t)")

        ignore_mask = y != self.module.ignore_label
        y = y[ignore_mask]

        preds = DUEMixin.forward(self.module, X, ignore_mask=ignore_mask, **kwargs)
        loss = -self.module.loss_function(preds, y)

        return loss
