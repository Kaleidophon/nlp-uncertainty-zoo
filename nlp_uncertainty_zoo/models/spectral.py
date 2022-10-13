"""
Implement a superclass for transformer models that use spectral normalization to meet the bi-Lipschitz condition.
The following models inherit from this class:

* Spectral-normalized Gaussian Process (SNGP) Transformer (`Liu et al., 2020 <https://arxiv.org/pdf/2006.10108.pdf>`)
* Deterministic Uncertainty Estimation (DUE) Transformer
(`Van Amersfoort et al., 2021 <https://arxiv.org/pdf/2102.11409.pdf>`)
* Deep Deterministic Uncertainty (DDU) Transformer (`Mukhoti et al., 2021 <https://arxiv.org/pdf/2102.11582.pdf>`)

Utility functions from the repository of Joost van Amsterfoort (https://github.com/y0ast/DUE) copied in here to avoid
import issues, since the original repository is not available as a Python package.
"""

# EXT
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)

# PROJECT
from nlp_uncertainty_zoo.models.bert import BertModule
from nlp_uncertainty_zoo.models.transformer import TransformerModule
from nlp_uncertainty_zoo.utils.custom_types import Device


class SpectralNormFC(SpectralNorm):
    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(
                        torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
                    )
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log = getattr(module, self.name + "_sigma")
        sigma_log.copy_(sigma.detach())

        return weight

    @staticmethod
    def apply(
        module: nn.Module,
        coeff: float,
        name: str,
        n_power_iterations: int,
        dim: int,
        eps: float,
    ) -> "SpectralNormFC":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNormFC(name, n_power_iterations, dim, eps)
        fn.coeff = coeff

        weight = module._parameters[name]
        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1))

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def spectral_norm_fc(
    module,
    coeff: float,
    n_power_iterations: int = 1,
    name: str = "weight",
    eps: float = 1e-12,
    dim: int = None,
):
    """
    Args:
        module (nn.Module): containing module
        coeff (float, optional): coefficient to normalize to
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        name (str, optional): name of weight parameter
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``
    Returns:
        The original module with the spectral norm hook
    Example::
        >>> m = spectral_norm_fc(nn.Linear(20, 40), 2.0)
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])
    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    SpectralNormFC.apply(module, coeff, name, n_power_iterations, dim, eps)
    return module


class SpectralTransformerModule(TransformerModule):
    """
    Implementation of a spectrally-normalized transformer. Used as a base for models like SNGP and DDU.
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
        Initialize a spectrally-normalized transformer.

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
            Dropout on word embeddings.
        dropout: float
            Dropout rate.
        num_heads: int
            Number of self-attention heads per layer.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings.
        spectral_norm_upper_bound: float
            Set a limit when weight matrices will be spectrally normalized if their eigenvalue surpasses it.
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


class SpectralBertModule(BertModule):
    """
    Implementation of a BERT model that uses spectral normalization.
    """

    def __init__(
        self,
        bert_name: str,
        output_size: int,
        spectral_norm_upper_bound: float,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize a spectrally-normalized BERT.

        Parameters
        ----------
        bert_name: str
            Name of the underlying BERT, as specified in HuggingFace transformers.
        output_size: int
            Number of classes.
        spectral_norm_upper_bound: float
            Set a limit when weight matrices will be spectrally normalized if their eigenvalue surpasses it.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            Device the model should be moved to.
        """
        super().__init__(
            bert_name, output_size, is_sequence_classifier, device, **build_params
        )

        self.spectral_norm_upper_bound = spectral_norm_upper_bound

        # Add spectral normalization
        self.output = spectral_norm_fc(self.output, coeff=spectral_norm_upper_bound)
        self.bert.pooler.dense = spectral_norm_fc(
            self.bert.pooler.dense, coeff=spectral_norm_upper_bound
        )

        # Since Bert module are stored in an OrderedDict which is not mutable, so we simply create a new module dict
        # and add spectral norm to Linear layers this way.
        for module_name, module in self.bert.encoder.named_modules():
            if isinstance(module, nn.Linear):
                setattr(
                    self.bert.encoder,
                    module_name,
                    spectral_norm_fc(module, coeff=spectral_norm_upper_bound),
                )
