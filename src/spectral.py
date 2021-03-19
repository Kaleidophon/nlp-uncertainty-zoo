"""
Implement transformer models that use spectral normalization to meet the bi-Lipschitz condition. More precisely,
this module implements a mixin enabling spectral normalization and, inheriting from that, the following two models:

* Spectral-normalized Gaussian Process (SNGP) Transformer (`Liu et al., 2020 <https://arxiv.org/pdf/2006.10108.pdf>`)
* Deep Deterministic Uncertainty (DDU) Transformer (`Mukhoti et al., 2021 <https://arxiv.org/pdf/2102.11582.pdf>`)
"""

# PROJECT
from src.transformer import Transformer, TransformerModule


class SpectralNormMixin:
    ...  # TODO


class SNGPTransformer(Transformer, SpectralNormMixin):
    ...  # TODO


class SNGPTransformerModule(TransformerModule):
    ...  # TODO


class DDUTransformer(Transformer, SpectralNormMixin):
    ...  # TODO


class DDUTransformerModule(TransformerModule):
    ...  # TODO
