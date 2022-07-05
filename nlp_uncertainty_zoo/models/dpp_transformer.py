"""
Implementing MC Dropout estimates using Determinantal Point Processes by Shelmanov et al. (2021) [1]. Code is a modified
version of their codebase (https://github.com/skoltech-nlp/certain-transformer).

[1] https://aclanthology.org/2021.eacl-main.157.pdf
"""

# STD
from typing import Dict, Any, Optional

# EXT
from alpaca.uncertainty_estimator.masks import build_mask
import torch
import torch.nn.functional as F

# PROJECT
from nlp_uncertainty_zoo.models.variational_transformer import VariationalBertModule
from nlp_uncertainty_zoo.models.model import Model
from nlp_uncertainty_zoo.utils.custom_types import Device


class DropoutDPP(torch.nn.Module):
    def __init__(
        self,
        p: float,
        activate=False,
        mask_name="dpp",
        max_n=100,
        max_frac=0.4,
        coef=1.0,
    ):
        super().__init__()

        self.activate = activate
        self.p = p
        self.p_init = p

        self.mask = build_mask(mask_name)
        self.reset_mask = False
        self.max_n = max_n
        self.max_frac = max_frac
        self.coef = coef

    def forward(self, x: torch.Tensor):
        if self.training:
            return F.dropout(x, self.p, training=True)
        else:
            if not self.activate:
                return x

            sum_mask = self.get_mask(x)

            i = 1
            frac_nonzero = self.calc_non_zero_neurons(sum_mask)
            while i < self.max_n and frac_nonzero < self.max_frac:
                mask = self.get_mask(x)

                sum_mask += mask
                i += 1

                frac_nonzero = self.calc_non_zero_neurons(sum_mask)

            res = x * sum_mask / i

            return res

    def get_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    def calc_non_zero_neurons(self, sum_mask):
        frac_nonzero = (sum_mask != 0).sum(axis=-1).item() / sum_mask.shape[-1]
        return frac_nonzero


class DPPBertModule(VariationalBertModule):
    """
    Implementation of Variational Transformer by `Xiao et al., (2021) <https://arxiv.org/pdf/2006.08344.pdf>`_ for BERT.
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
            Name of the BERT to be used.
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


class DPPBert(Model):
    """
    Variational version of BERT.
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
