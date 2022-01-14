"""
Implement a superclass for transformer models that use spectral normalization to meet the bi-Lipschitz condition.
The following models inherit from this class:

* Spectral-normalized Gaussian Process (SNGP) Transformer (`Liu et al., 2020 <https://arxiv.org/pdf/2006.10108.pdf>`)
* Deterministic Uncertainty Estimation (DUE) Transformer
(`Van Amersfoort et al., 2021 <https://arxiv.org/pdf/2102.11409.pdf>`)
* Deep Deterministic Uncertainty (DDU) Transformer (`Mukhoti et al., 2021 <https://arxiv.org/pdf/2102.11582.pdf>`)
"""

# EXT
from due.layers.spectral_norm_fc import spectral_norm_fc
import torch.nn as nn

# PROJECT
from nlp_uncertainty_zoo.models.transformer import TransformerModule
from nlp_uncertainty_zoo.utils.custom_types import Device


# TODO: Add spectral Bert


class SpectralTransformerModule(TransformerModule):
    """
    Implementation of a spectral-normalized transformer. Used as a base for models like SNGP, DUE and DDU.
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
            is_sequence_classifier,
            device,
        )

        # Add spectral normalization
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                setattr(
                    self,
                    module_name,
                    spectral_norm_fc(module, coeff=spectral_norm_upper_bound),
                )
