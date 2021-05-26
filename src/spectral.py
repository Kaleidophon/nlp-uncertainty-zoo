"""
Implement transformer models that use spectral normalization to meet the bi-Lipschitz condition. More precisely,
this module implements a mixin enabling spectral normalization and, inheriting from that, the following two models:

* Spectral-normalized Gaussian Process (SNGP) Transformer (`Liu et al., 2020 <https://arxiv.org/pdf/2006.10108.pdf>`)
* Deep Deterministic Uncertainty (DDU) Transformer (`Mukhoti et al., 2021 <https://arxiv.org/pdf/2102.11582.pdf>`)
"""

# STD
import math

# EXT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Any, Optional

# PROJECT
from src.datasets import DataSplit
from src.transformer import TransformerModule
from src.model import Model
from src.types import Device


class SNGPOutputModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        scaling_coefficient: float,
        beta_length_scale: float,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.scaling_coefficent = scaling_coefficient
        self.beta_length_scale = beta_length_scale

        # ### Init parameters

        # Random, frozen output layer
        self.output = nn.Linear(self.hidden_size, self.output_size)
        # Change init of weights and biases following Liu et al. (2020)
        self.output.weight.data.normal_(0, 1)
        self.output.bias.data.uniform_(0, 2 * math.pi)

        # This layer is frozen right after init
        self.output.weight.requires_grad = False
        self.output.bias.requires_grad = False

        # Bundle all beta_k vectors in a matrix
        self.register_parameter(
            name="Beta",
            param=nn.Parameter(
                torch.randn(output_size, output_size) * beta_length_scale
            ),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        Phi = math.sqrt(2 / self.output_size) * torch.cos(
            self.output(-x)
        )  # Gives a output_size x 1 vector
        out = self.Beta @ Phi  # output_size x 1

        return out


class SNGPTransformerModule(TransformerModule):
    """
    Implementation of a spectral-normalized Gaussian Process transformer.
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        num_heads: int,
        sequence_length: int,
        spectral_norm_upper_bound: float,
        scaling_coefficent: float,
        beta_length_scale: float,
        device: Device,
    ):
        """
        Initialize a transformer.

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
        dropout: float
            Dropout rate.
        num_heads: int
            Number of self-attention heads per layer.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings.
        device: Device
            Device the model is located on.
        """
        super().__init__(
            num_layers,
            vocab_size,
            input_size,
            hidden_size,
            output_size,
            dropout,
            num_heads,
            sequence_length,
            device,
        )

        # TODO: Incorporate upper bound
        for module in self._modules.values():
            if isinstance(module, nn.Linear):
                utils.spectral_norm(module)  # Add spectral normalization

    def forward(self, input_: torch.LongTensor) -> torch.FloatTensor:
        word_embeddings = self.word_embeddings(input_)
        embeddings = self.pos_embeddings(word_embeddings)
        embeddings = self.input_dropout(embeddings)

        out = self.encoder(embeddings)
        out = self.output_dropout(out)

        return out


class SNGPTransformer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "sngp_transformer",
            SNGPTransformerModule,
            model_params,
            train_params,
            model_dir,
            device,
        )

        default_model_params = [
            param for name, param in self.module.named_parameters() if name != "Beta"
        ]
        self.optimizer = torch.optim.Adam(
            [
                {"params": default_model_params, "lr": train_params["lr"]},
                {
                    "params": [self.module.Beta],
                    "lr": train_params["lr"],
                    "weight_decay": train_params["weight_decay"],
                },
            ]
        )


class DDUTransformerModule(TransformerModule):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        num_heads: int,
        sequence_length: int,
        device: Device,
    ):
        """
        Initialize a transformer.

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
        dropout: float
            Dropout rate.
        num_heads: int
            Number of self-attention heads per layer.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings.
        device: Device
            Device the model is located on.
        """
        super().__init__(
            num_layers,
            vocab_size,
            input_size,
            hidden_size,
            output_size,
            dropout,
            num_heads,
            sequence_length,
            device,
        )

        # Parameters for Gaussian Discriminant Analysis
        self.mu = torch.zeros(output_size, hidden_size)
        self.Sigma = torch.zeros(output_size, hidden_size, hidden_size)
        self.num_classes = torch.zeros(output_size)


class DDUTransformer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "ddu_transformer",
            DDUTransformerModule,
            model_params,
            train_params,
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
        progress_bar = tqdm(total=len(data_split)) if verbose else None
        self.module.eval()  # Disable dropout

        with torch.no_grad():
            for i, (X, y) in enumerate(data_split):

                hidden = self.module._get_hidden(X)
                hidden, y = torch.flatten(hidden, end_dim=1), torch.flatten(y)

                for cls in y.unique():
                    num_batch_classes = (y == cls).long().sum()
                    self.module.num_classes[cls] += num_batch_classes
                    self.module.mu[cls] += hidden[y == cls].sum(dim=0)
                    self.module.Sigma[cls] += torch.FloatTensor(
                        np.cov(hidden[y == cls].T.numpy())
                    ) * (num_batch_classes - 1)

                if verbose:
                    progress_bar.set_description(
                        f"Fitting GDA (Batch {i+1}/{len(data_split)})"
                    )
                    progress_bar.update(1)

            self.module.mu = self.module.mu / self.module.num_classes.unsqueeze(1)
            self.module.Sigma = self.module.Sigma / self.module.num_classes.sum()
