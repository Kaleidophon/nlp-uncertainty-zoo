"""
Implementation of Deterministic Uncertainty Estimation (DUE) Transformer by
`Van Amersfoort et al., 2021 <https://arxiv.org/pdf/2102.11409.pdf>`_.
"""

# STD
import math
from typing import Dict, Any, Optional

# EXT
import torch
from due.dkl import _get_initial_inducing_points, _get_initial_lengthscale, GP
from einops import rearrange
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter

# PROJECT
from nlp_uncertainty_zoo.models.spectral import SpectralTransformerModule
from nlp_uncertainty_zoo.datasets import DataSplit, TextDataset
from nlp_uncertainty_zoo.models.model import MultiPredictionMixin, Model
from nlp_uncertainty_zoo.utils.types import Device


class DUETransformerModule(SpectralTransformerModule, MultiPredictionMixin):
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

        self.num_inducing_samples = num_inducing_samples
        self.num_inducing_points = num_inducing_points
        self.spectral_norm_upper_bound = spectral_norm_upper_bound
        self.kernel_type = kernel_type
        self.layer_norm = nn.LayerNorm([input_size])

        self.gp = None
        self.likelihood = None
        self.loss_function = None

    def init_gp(self, train_data: DataSplit, num_instances: int = 1000):
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
        batch_size = train_data[0][0].shape[0]
        num_batches = math.ceil(num_instances / (batch_size * self.sequence_length))

        # Essentially do the same as in due.dkl.initial_values_for_GP, but with a sequential dataset
        sampled_batch_idx = torch.randperm(len(train_data))[:num_batches]

        # Extract feature representations for sampled batches
        batch_representations = []

        with torch.no_grad():
            for batch_idx in sampled_batch_idx:
                X = train_data[batch_idx][0].to(self.device)
                batch_representations.append(self.get_hidden(X))

        representations = torch.cat(batch_representations, dim=0)
        representations = rearrange(representations, "b s h -> (b s) h")

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
            self.likelihood, self.gp, num_data=len(train_data)
        ).to(self.device)

    def forward(self, input_: torch.LongTensor):
        out = self.get_hidden(input_)

        if self.is_sequence_classifier:
            out = self.get_sequence_representation(out)

        out = self.layer_norm(out)
        out = rearrange(out, "b s h -> (b s) h").float()
        mvn = self.gp(out)

        return mvn

    def get_logits(self, input_: torch.LongTensor) -> torch.FloatTensor:
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
        batch_size = input_.shape[0]

        mvn = self.forward(input_)
        predictions = mvn.sample(sample_shape=torch.Size((self.num_predictions,)))
        predictions = rearrange(
            predictions, "n (b s) o  -> b n s o", b=batch_size, n=self.num_predictions
        )

        return predictions


class DUETransformer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "ddu_transformer",
            DUETransformerModule,
            model_params,
            train_params,
            model_dir,
            device,
        )

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
        # Retrieve inducing points and length scale from training set to initialize the GP
        self.module.init_gp(dataset.train)

        return super().fit(dataset, validate, verbose, summary_writer)

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        with torch.no_grad():
            out = self.module(X)
            out = self.module.likelihood(out)
            out = out.logits.mean(dim=0)

        return out

    def get_loss(
        self,
        n_batch: int,
        X: torch.Tensor,
        y: torch.Tensor,
        summary_writer: Optional[SummaryWriter] = None,
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
        summary_writer: SummaryWriter
            Summary writer to track training statistics.

        Returns
        -------
        torch.Tensor
            Batch loss.
        """
        # TODO: This needs to be adapted for language modelling
        preds = self.module(X)
        loss = -self.module.loss_function(preds, y)

        return loss
