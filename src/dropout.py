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
from src.lstm import LSTMModule
from src.model import Model
from src.transformer import TransformerModule
from src.types import Device


class VariationalLSTMModule(LSTMModule):
    """
    Implementation of variational LSTM by `(Gal & Ghrahramani, 2016b) <https://arxiv.org/pdf/1512.05287.pdf>`.
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
        num_predictions: int,
        device: Device,
    ):
        """
        Initialize a LSTM.

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
            Dropout on word embeddings. Dropout application corresponds to `Gal & Ghahramani (2016)
            <https://papers.nips.cc/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf>`_.
        dropout: float
            Dropout rate. Dropout application corresponds to `Gal & Ghahramani (2016)
            <https://papers.nips.cc/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf>`_.
        num_predictions: int
            Number of predictions with different dropout masks.
        device: Device
            Device the model is located on.
        """
        self.num_predictions = num_predictions

        super().__init__(
            num_layers,
            vocab_size,
            input_size,
            hidden_size,
            output_size,
            input_dropout,
            dropout,
            device,
        )


class VariationalLSTM(Model):
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
            VariationalLSTMModule,
            model_params,
            train_params,
            model_dir,
            device,
        )

    def predict(
        self, X: torch.Tensor, num_predictions: Optional[int] = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Make a prediction for some input.

        Parameters
        ----------
        X: torch.Tensor
            Input data points.
        num_predictions: int
            Number of predictions. In this case, equivalent to multiple forward passes with different dropout masks.
            If None, the attribute of the same name set during initialization is used.

        Returns
        -------
        torch.Tensor
            Predictions.
        """
        if num_predictions is None:
            num_predictions = self.module.num_predictions

        X = X.to(self.device)

        batch, seq_len = X.shape
        preds = torch.zeros(batch, seq_len, self.module.output_size, device=self.device)

        for _ in range(num_predictions):
            preds += self.module(X)

        preds /= num_predictions

        return preds


class VariationalTransformerModule(TransformerModule):
    """
    Implementation of Variational Transformer by `Xiao et al., (2021) <https://arxiv.org/pdf/2006.08344.pdf>`_.
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
        input_dropout: float
            Dropout on word embeddings. Dropout application corresponds to `Gal & Ghahramani (2016)
            <https://papers.nips.cc/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf>`_.
        dropout: float
            Dropout rate.
        num_heads: int
            Number of self-attention heads per layer.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings.
        num_predictions: int
            Number of predictions with different dropout masks.
        device: Device
            Device the model is located on.
        """

        self.num_predictions = num_predictions

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
            device,
        )

    def eval(self, *args):
        super().eval()

        for module in self._modules.values():
            if isinstance(module, torch.nn.Dropout):
                module.train()


class VariationalTransformer(Model):
    """
    Module for the variational transformer.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "variational_transformer",
            VariationalTransformerModule,
            model_params,
            train_params,
            model_dir,
            device,
        )

    def predict(
        self, X: torch.Tensor, num_predictions: Optional[int] = None, *args, **kwargs
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
        if num_predictions is None:
            num_predictions = self.module.num_predictions

        X = X.to(self.device)

        batch, seq_len = X.shape
        preds = torch.zeros(batch, seq_len, self.module.output_size, device=self.device)

        for _ in range(num_predictions):
            preds += self.module(X)

        preds /= num_predictions

        return preds
