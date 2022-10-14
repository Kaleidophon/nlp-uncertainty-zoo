"""
Implementing MC Dropout estimates using Determinantal Point Processes by
`Shelmanov et al. (2021) <https://aclanthology.org/2021.eacl-main.157.pdf>_`. Code is a modified version of
`their codebase <https://github.com/skoltech-nlp/certain-transformer>_`.
"""

# STD
from typing import Dict, Any, Optional

# EXT
from alpaca.uncertainty_estimator.masks import build_mask
from einops import rearrange
import torch
import torch.nn.functional as F

# PROJECT
from nlp_uncertainty_zoo.models.variational_transformer import VariationalBertModule, VariationalTransformerModule
from nlp_uncertainty_zoo.models.model import Model
from nlp_uncertainty_zoo.utils.custom_types import Device


class DropoutDPP(torch.nn.Module):
    """
    Implementation of the determinantal point process dropout, modified version of
    `the original implementation <https://github.com/skoltech-nlp/certain-transformer/blob/main/src/ue4nlp/dropout_dpp.py>_`.
    """
    dropout_id = -1

    def __init__(
        self,
        p: float,
        max_n: int = 100,
        max_frac: float = 0.4,

    ):
        """
        Initialize a DPP dropout module.

        Parameters
        ----------
        p: float
            Dropout probability.
        max_n: int
            Maximum number of iterations.
        max_frac: float
            Maximum fraction of dropped-out neurons.
        """
        super().__init__()

        self.p = p

        self.mask = build_mask("dpp")
        self.max_n = max_n
        self.max_frac = max_frac

        self.curr_dropout_id = DropoutDPP.update()

    @classmethod
    def update(cls):
        cls.dropout_id += 1
        return cls.dropout_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to the input. If model is in training mode, apply a normal dropout mask. If the model is in
        inference mode, apply DPP dropout.

        Parameters
        ----------
        x: torch.Tensor
            Input to which dropout mask is being applied to.

        Returns
        -------
        torch.Tensor
            Input after applying dropout mask.
        """
        if self.training:
            return F.dropout(x, self.p, training=True)

        else:
            # For attention weights
            if len(x.shape) == 4:
                return F.dropout(x, self.p, training=True)

            batch_size = x.shape[0]
            x = rearrange(x, "b t p -> (b t) p")
            sum_mask = self.get_mask(x)

            i = 1
            frac_nonzero = self.calc_non_zero_neurons(sum_mask)

            while i < self.max_n and frac_nonzero < self.max_frac:
                mask = self.get_mask(x)

                sum_mask += mask
                i += 1

                frac_nonzero = self.calc_non_zero_neurons(sum_mask)

            res = x * sum_mask / i
            res = rearrange(res, "(b t) p -> b t p", b=batch_size)

            return res

    def get_mask(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Generate a new mask for a given input.

        Parameters
        ----------
        x: torch.Tensor
            Input for which the mask is supposed to be generated.

        Returns
        -------
        torch.FloatTensor

        """
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    @staticmethod
    def calc_non_zero_neurons(sum_mask) -> float:
        """
        Compute the fraction of neurons that have not been dropped out yet.

        Parameters
        ----------
        sum_mask: torch.Tensor
            Current mask.

        Returns
        -------
        float
            Percentage of non-dropped out neurons.
        """
        frac_nonzero = (sum_mask != 0).sum(axis=-1).item() / sum_mask.shape[-1]
        return frac_nonzero


class DPPTransformerModule(VariationalTransformerModule):
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
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
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
        input_dropout: float
            Dropout on word embeddings.
        dropout: float
            Dropout rate.
        num_heads: int
            Number of self-attention heads per layer.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings.
        num_predictions: int
            Number of predictions with different dropout masks.
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
            num_predictions,
            is_sequence_classifier,
            device,
            **build_params
        )

    def eval(self, *args):
        # For the superclass VariationalTransformerModule, all dropout modules will still be set to training mode, even
        # when the rest of the model is in inference mode. Here, we actually want all of the model to be in inference
        # mode for the DPPDropout modules to work correctly. Therefore, revert behavior to the original torch Module
        # implementation.
        torch.nn.Module.eval(self)


class DPPTransformer(Model):
    """
    Transformer model using determinantal point process dropout.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            f"dpp-transformer",
            DPPTransformerModule,
            model_params,
            model_dir,
            device,
        )


class DPPBertModule(VariationalBertModule):
    """
    Implementation of a transformer with determinantal point process dropout for BERT.
    """

    def __init__(
        self,
        bert_name: str,
        output_size: int,
        dropout: float,
        num_predictions: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize a transformer.

        Parameters
        ----------
        bert_name: str
            Name of the underlying BERT, as specified in HuggingFace transformers.
        dropout: float
            Dropout probability.
        num_predictions: int
            Number of predictions with different dropout masks.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            Device the model is located on.
        """
        self.num_predictions = num_predictions

        super().__init__(
            bert_name,
            output_size,
            dropout,
            num_predictions,
            is_sequence_classifier,
            device,
        )

        # Replace all dropout layers with DPP dropout
        for name, module in list(self.named_modules()):
            if isinstance(module, torch.nn.Dropout):
                dpp_module = DropoutDPP(p=dropout)
                sub_objs = name.split(".")
                target_obj = sub_objs[-1]
                current_obj = self

                for obj in sub_objs[:-1]:
                    current_obj = getattr(current_obj, obj)

                setattr(current_obj, target_obj, dpp_module)

    def eval(self, *args):
        # For the superclass VariationalBertModule, all dropout modules will still be set to training mode, even when
        # the rest of the model is in inference mode. Here, we actually want all of the model to be in inference mode
        # for the DPPDropout modules to work correctly. Therefore, revert behavior to the original torch Module
        # implementation.
        torch.nn.Module.eval(self)


class DPPBert(Model):
    """
    BERT model using determinantal point process dropout.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        bert_name = model_params["bert_name"]
        super().__init__(
            f"dpp-{bert_name}",
            DPPBertModule,
            model_params,
            model_dir,
            device,
        )
