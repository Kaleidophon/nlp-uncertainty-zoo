"""
Implement a simple mixin that allows for inference using MC Dropout `Gal & Ghrahramani (2016a) `
<http://proceedings.mlr.press/v48/gal16.pdf> and corresponding subclasses of the LSTM and transformer, creating two
models:

* Variational LSTM `(Gal & Ghrahramani, 2016b) <https://arxiv.org/pdf/1512.05287.pdf>`
* Variational Transformer `(Xiao et al., 2021) <https://arxiv.org/pdf/2006.08344.pdf>`
"""

# STD
from typing import Dict, Any, Optional

# EXT
import torch

# PROJECT
from src.lstm import LSTM
from src.module import Module, Model
from src.types import Device


class VariationalLSTM(LSTM):
    """
    Implementation of variational LSTM by `(Gal & Ghrahramani, 2016b) <https://arxiv.org/pdf/1512.05287.pdf>`.
    """

    def train(self, *args):
        # By calling the grandparent method here, we use dropout even during inference
        Model.train(self, *args)

    def eval(self):
        Model.eval(self)


class VariationalLSTMModule(Module):
    """
    Module for the variational LSTM.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "variational_lstm",
            VariationalLSTM,
            model_params,
            train_params,
            model_dir,
            device,
        )

    def predict(
        self, X: torch.Tensor, num_predictions: int = 10, *args, **kwargs
    ) -> torch.Tensor:
        """
        Make a prediction for some input.

        Parameters
        ----------
        X: torch.Tensor
            Input data points.
        num_predictions: int
            Number of predictions. In this case, equivalent to multiple forward passes with different dropout masks.

        Returns
        -------
        torch.Tensor
            Predictions.
        """
        X.to(self.device)

        preds = []
        for _ in range(num_predictions):
            preds.append(self.model(X))

        preds = torch.stack(preds, dim=1)

        return preds


class VariationalTransformer:
    ...  # TODO


class VariationalTransformerModule(Module):
    ...  # TODO
