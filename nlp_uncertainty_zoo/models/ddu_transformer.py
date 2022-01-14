"""
Implementation of the Deep Deterministic Uncertainty (DDU) Transformer by
s`Mukhoti et al. (2021) <https://arxiv.org/pdf/2102.11582.pdf>`_.
"""

# STD
import math
from typing import Optional, Dict, Any

# EXT
import numpy as np
import torch
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

# PROJECT
from nlp_uncertainty_zoo.models.spectral import SpectralTransformerModule
from nlp_uncertainty_zoo.datasets import DataSplit
from nlp_uncertainty_zoo.models.model import Model
from nlp_uncertainty_zoo.utils.custom_types import Device


# TODO: Write version of this which accepts a pre-trained model that is to be fine-tuned


class DDUTransformerModule(SpectralTransformerModule):
    """
    Implementation of the Deep Deterministic Uncertainty (DDU) Transformer by
    `Mukhoti et al. (2021) <https://arxiv.org/pdf/2102.11582.pdf>`_.
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
        spectral_norm_upper_bound: float,
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

        # Parameters for Gaussian Discriminant Analysis
        self.mu = torch.zeros(output_size, input_size)
        self.Sigma = torch.stack(
            [torch.eye(input_size, input_size) for _ in range(self.output_size)]
        )
        self.determinants = torch.zeros(output_size)

    def gmm_fit(self, data_split: DataSplit) -> None:
        """
        Fit a Gaussian mixture model on the feature representations of the trained model.

        Parameters
        ----------
        data_split: DataSplit
            Data split used for fitting, usually the training or validation split.
        """
        with torch.no_grad():
            hiddens, labels = [], []

            for i, (X, y) in enumerate(data_split):

                hidden = self.get_hidden(X)
                hidden = (
                    self.get_sequence_representation(hidden).squeeze(1)
                    if self.is_sequence_classifier
                    else torch.flatten(hidden, end_dim=1)
                )
                y = torch.flatten(y)
                hiddens.append(hidden)
                labels.append(y)

            hiddens = torch.cat(hiddens, dim=0)
            labels = torch.cat(labels, dim=0)

            for cls in labels.unique():
                num_batch_classes = (labels == cls).long().sum()

                if num_batch_classes == 0:
                    continue

                self.mu[cls] = hiddens[labels == cls].mean(dim=0)
                self.Sigma[cls] = torch.FloatTensor(
                    np.cov(hiddens[labels == cls].T.numpy())
                ) * (num_batch_classes - 1)

                self.determinants[cls] = torch.det(
                    self.Sigma[cls, :, :]
                )  # Compute determinant
                self.Sigma[cls, :, :] = torch.linalg.inv(self.Sigma[cls, :, :])

    def gmm_predict(self, input_: torch.LongTensor) -> torch.FloatTensor:
        """
        Make a prediction with the Gaussian mixture Model for a batch of inputs.

        Parameters
        ----------
        input_: torch.LongTensor
            Batch of inputs in the form of indexed tokens.

        Returns
        -------
        torch.FloatTensor
            Probability of the input under every mixture component, with one component per class.
        """
        batch_size = input_.shape[0]
        hidden = self.get_hidden(input_)  # batch_size x seq_length x input_size
        hidden = (
            self.get_sequence_representation(hidden).squeeze(1)
            if self.is_sequence_classifier
            else rearrange(hidden, "b s i -> (b s) i")
        )

        hidden = hidden.unsqueeze(1)  # (batch_size x seq_length) x 1 x input_size
        hidden = hidden.repeat(1, self.output_size, 1)
        diff = hidden - self.mu  # (batch_size x seq_length) x output_size x input_size
        diff_t = rearrange(diff, "b o i -> b i o")

        probs = (
            1
            / (2 * math.pi * self.determinants + 1e-6)
            * torch.exp(
                -0.5 * torch.einsum("boi,oii,bio->bo", diff, self.Sigma, diff_t)
            )
        )  # (batch_size x seq_length) x output_size
        probs = rearrange(probs, "(b s) o -> b s o", b=batch_size)

        return probs

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
        metric_name: str
            Name of uncertainty metric being used.

        Returns
        -------
        torch.FloatTensor
            Uncertainty scores for the current batch.
        """
        if metric_name is None:
            metric_name = self.default_uncertainty_metric

        if metric_name == "log_prob":
            with torch.no_grad():
                return self.gmm_predict(input_)

        else:
            return super().get_uncertainty(input_, metric_name)


class DDUTransformer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "ddu_transformer",
            DDUTransformerModule,
            model_params,
            model_dir,
            device,
        )

    def _finetune(
        self,
        data_split: DataSplit,
        verbose: bool,
        summary_writer: Optional[SummaryWriter] = None,
    ):
        """
        As an additional step after training, DDU fits a Gaussian Discriminant Analysis model to
        the training data.

        Parameters
        ----------
        data_split: DataSplit
            Data the GDA is fit on.
        verbose: bool
            Whether to display information about current loss.
        summary_writer: Optional[SummaryWriter]
            Summary writer to track training statistics. Training and validation loss (if applicable) are tracked by
            default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        self.module.eval()  # Disable dropout
        self.module.gmm_fit(data_split)
        self.module.train()
