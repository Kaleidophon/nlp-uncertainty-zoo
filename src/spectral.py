"""
Implement transformer models that use spectral normalization to meet the bi-Lipschitz condition. More precisely,
this module implements a mixin enabling spectral normalization and, inheriting from that, the following two models:

* Spectral-normalized Gaussian Process (SNGP) Transformer (`Liu et al., 2020 <https://arxiv.org/pdf/2006.10108.pdf>`)
* Deep Deterministic Uncertainty (DDU) Transformer (`Mukhoti et al., 2021 <https://arxiv.org/pdf/2102.11582.pdf>`)
"""

# STD
import math

# EXT
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional

# PROJECT
from src.transformer import TransformerModule
from src.model import Model
from src.types import Device


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

        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        # Change init of weights and biases following Liu et al. (2020)
        self.output.weight.data.normal_(0, 1)
        self.output.bias.data.uniform_(0, 2 * math.pi)

        self.register_parameter(
            name="Beta", param=nn.Parameter(torch.randn(hidden_size, output_size))
        )

        for module in self._modules.values():
            if isinstance(module, nn.Linear):
                utils.spectral_norm(module)  # Add spectral normalization

    def forward(self, input_: torch.LongTensor) -> torch.FloatTensor:
        word_embeddings = self.word_embeddings(input_)
        embeddings = self.pos_embeddings(word_embeddings)
        embeddings = self.input_dropout(embeddings)

        out = self.encoder(embeddings)
        out = self.output_dropout(out)
        Phi = math.sqrt(2 / self.hidden_size) * torch.cos(self.output(-out))
        out = Phi @ self.Beta

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
    ...  # TODO


class DDUTransformer(Model):
    ...  # TODO
