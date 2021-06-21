"""
Implement a simple mixin that allows for inference using MC Dropout `Gal & Ghrahramani (2016a) `
<http://proceedings.mlr.press/v48/gal16.pdf> and corresponding subclasses of the LSTM and transformer, creating two
models:

* Variational LSTM `(Gal & Ghrahramani, 2016b) <https://arxiv.org/pdf/1512.05287.pdf>`
* Variational Transformer `(Xiao et al., 2021) <https://arxiv.org/pdf/2006.08344.pdf>`
"""

# STD
from typing import Dict, Any, Optional, Tuple

# EXT
import torch
import torch.nn as nn
import torch.nn.functional as F

# PROJECT
from src.model import Model
from src.transformer import TransformerModule
from src.types import Device, HiddenDict

# TODO: Add missing docstrings


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define gates
        self.x_input = nn.Linear(input_size, hidden_size)
        self.x_forget = nn.Linear(input_size, hidden_size)
        self.x_cell = nn.Linear(input_size, hidden_size)
        self.x_output = nn.Linear(input_size, hidden_size)
        # Only init one bias term per layer
        self.h_input = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_forget = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_cell = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.FloatTensor,
        hx: torch.FloatTensor,
        cx: torch.FloatTensor,
        mask_x: torch.FloatTensor,
        mask_h: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        mask_fx, mask_ix, mask_ox, mask_cx = torch.split(mask_x, self.input_size, dim=1)
        mask_fh, mask_ih, mask_oh, mask_ch = torch.split(
            mask_h, self.hidden_size, dim=1
        )

        # Forget gate
        f_g = torch.sigmoid(self.x_forget(x * mask_fx) + self.h_forget(hx * mask_fh))

        # Input gate
        i_g = torch.sigmoid(self.x_input(x * mask_ix) + self.h_input(hx * mask_ih))

        # Output gate
        o_g = torch.sigmoid(self.x_output(x * mask_ox) + self.h_output(hx * mask_oh))

        # Intermediate cell state
        c_tilde_g = torch.tanh(self.x_cell(x * mask_cx) + self.h_cell(x * mask_ch))

        # New cell state
        cx = f_g * cx + i_g * c_tilde_g

        # New hidden state
        hx = o_g * torch.tanh(cx)

        return hx, cx


class VariationalDropout:
    def __init__(self, dropout: float, input_dim: int, device: Device):
        self.dropout = dropout
        self.input_dim = input_dim
        self.device = device
        self.mask = None

    def sample(self, batch_size: int):
        self.mask = torch.bernoulli(
            torch.ones(batch_size, self.input_dim, device=self.device)
            * (1 - self.dropout)
        ) / (1 - self.dropout)

        return self.mask


class VariationalLSTMModule(nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        embedding_dropout: float,
        layer_dropout: float,
        time_dropout: float,
        num_predictions: int,
        device: Device,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        self.lstm_layers = nn.ModuleList(
            [
                CustomLSTMCell(
                    in_size,
                    hidden_size,
                ).to(device)
                for in_size in [input_size] + [hidden_size] * (num_layers - 1)
            ]
        )
        self.embeddings = nn.Embedding(vocab_size, input_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.num_predictions = num_predictions

        self.dropout_modules = {
            "layer": [
                VariationalDropout(dropout, size, device)
                for dropout, size in zip(
                    [embedding_dropout]
                    + [layer_dropout]
                    * num_layers,  # Use different rate after embeddings
                    [hidden_size * 4] * num_layers
                    + [hidden_size],  # We don't have gates in the last layer
                )
            ],
            "time": [
                VariationalDropout(time_dropout, hidden_size * 4, device)
                for _ in range(num_layers)
            ],
        }

        self.last_hidden_states = None

    def forward(
        self,
        input_: torch.LongTensor,
        hidden_states: Optional[torch.FloatTensor] = None,
    ):
        batch_size, sequence_length = input_.shape
        outputs = []

        # Initialize hidden activations if not given
        if hidden_states is None and self.last_hidden_states is None:
            hidden_states = self.init_hidden_states(batch_size, self.device)
        # Detach hidden activations to limit gradient computations
        else:
            hidden_states = (
                self.last_hidden_states if hidden_states is None else hidden_states
            )

        # Sample dropout masks used throughout this batch
        dropout_masks = self.sample_masks(batch_size)

        for t in range(sequence_length):

            layer_input = self.embeddings(input_[:, t])
            # layer_input = self.dropout_modules["embedding"](embeddings, input_[:, t])

            for layer, cell in enumerate(self.lstm_layers):
                hx, cx = cell(
                    layer_input,  # Hidden state of last layer
                    *hidden_states[
                        layer
                    ],  # Hidden and cell state of previous time step
                    mask_x=dropout_masks["layer"][layer],
                    mask_h=dropout_masks["time"][layer]
                )
                layer_input = hx  # This becomes the input for the next layer
                hidden_states[layer] = (
                    hx,
                    cx,
                )  # This becomes the input for the next time step

            layer_input *= dropout_masks["layer"][self.num_layers]
            outputs.append(self.decoder(layer_input))

        outputs = torch.stack(outputs, dim=1)
        self._assign_last_hidden_states(hidden_states)

        return outputs

    # TODO: Add doc here
    def _assign_last_hidden_states(self, hidden: HiddenDict):
        self.last_hidden_states = {
            layer: (h[0].detach(), h[1].detach()) for layer, h in hidden.items()
        }

    # TODO: Add doc here
    def init_hidden_states(self, batch_size: int, device: Device) -> HiddenDict:
        hidden = {
            layer: (
                torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device),
            )
            for layer in range(self.num_layers)
        }

        return hidden

    def sample_masks(self, batch_size: int):
        return {
            dropout_type: {
                layer: layer_module.sample(batch_size)
                # Iterate over all dropout modules of one type (across different layers)
                for layer, layer_module in enumerate(dropout_modules)
            }
            # Iterate over type of dropout modules ("layer", "time")
            for dropout_type, dropout_modules in self.dropout_modules.items()
        }


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

        # Only for Gal & Ghahramani replication, I know this isn't pretty
        if "init_weight" in train_params:
            init_weight = train_params["init_weight"]

            for cell in self.module.lstm_layers:
                for param in cell.parameters():
                    param.data.uniform_(-init_weight, init_weight)

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

        batch_size, seq_len = X.shape
        preds = torch.zeros(
            batch_size, seq_len, self.module.output_size, device=self.device
        )

        # Make sure that the same hidden state from the last batch is used for all forward passes
        # Init hidden state - continue with hidden states from last batch
        hidden_states = self.module.last_hidden_states

        # This would e.g. happen when model is switched from train() to eval() - init hidden states with zeros
        if hidden_states is None:
            hidden_states = self.module.init_hidden_states(batch_size, self.device)

        with torch.no_grad():
            for _ in range(num_predictions):
                preds += F.softmax(self.module(X, hidden_states=hidden_states), dim=-1)

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

        with torch.no_grad():
            preds = torch.zeros(
                batch, seq_len, self.module.output_size, device=self.device
            )

            for _ in range(num_predictions):
                preds += self.module(X)

            preds /= num_predictions

        return preds
