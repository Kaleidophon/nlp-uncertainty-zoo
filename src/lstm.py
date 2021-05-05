"""
Implement a vanilla LSTM model.
"""

# STD
from typing import Optional, Dict, Any

# EXT
import torch
import torch.nn as nn

# PROJECT
from src.model import Model, Module
from src.types import HiddenDict, Device, HiddenStates


class LSTMModule(Module):
    """
    Implementation of a LSTM.
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
        device: Device
            Device the model is located on.
        """
        super().__init__(
            num_layers, vocab_size, input_size, hidden_size, output_size, device
        )

        # Initialize network
        self.embeddings = nn.Embedding(vocab_size, input_size)
        self.gates = {}
        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = dropout
        self.input_dropout = input_dropout
        self._dropout = 0
        self._input_dropout = input_dropout

        for layer in range(num_layers):
            self.gates[layer] = {
                "ii": nn.Linear(input_size, hidden_size).to(self.device),
                "if": nn.Linear(input_size, hidden_size).to(self.device),
                "ig": nn.Linear(input_size, hidden_size).to(self.device),
                "io": nn.Linear(input_size, hidden_size).to(self.device),
                "hi": nn.Linear(hidden_size, hidden_size).to(self.device),
                "hf": nn.Linear(hidden_size, hidden_size).to(self.device),
                "hg": nn.Linear(hidden_size, hidden_size).to(self.device),
                "ho": nn.Linear(hidden_size, hidden_size).to(self.device),
            }

    def forward(
        self, input_: torch.LongTensor, hidden: Optional[HiddenDict] = None
    ) -> torch.FloatTensor:
        batch_size, sequence_length = input_.shape

        # Initialize hidden activations if not given
        if hidden is None:
            hidden = {
                layer: (
                    torch.zeros(batch_size, self.hidden_size, device=self.device),
                    torch.zeros(batch_size, self.hidden_size, device=self.device),
                )
                for layer in range(self.num_layers)
            }

        # Sample all dropout masks used for this batch
        dropout_masks_input = {
            layer: torch.bernoulli(
                torch.ones(batch_size, self.hidden_size, device=self.device)
                * (1 - self.input_dropout)
            )
            for layer in range(self.num_layers)
        }
        dropout_masks_time = {
            layer: torch.bernoulli(
                torch.ones(batch_size, self.hidden_size, device=self.device)
                * (1 - self.dropout)
            )
            for layer in range(self.num_layers)
        }

        outputs = []

        for t in range(sequence_length):

            embeddings = self.embeddings(input_[:, t])
            layer_input = embeddings.squeeze(0)

            for layer in range(self.num_layers):
                new_hidden = self.forward_step(
                    layer,
                    hidden[layer],
                    layer_input,
                    dropout_masks_input[layer],
                    dropout_masks_time[layer],
                )
                layer_input = new_hidden[
                    0
                ]  # New hidden state becomes input for next layer
                hidden[layer] = new_hidden  # Store for next step

            dropout_out = torch.bernoulli(
                torch.ones(batch_size, self.hidden_size, device=self.device)
                * (1 - self.dropout)
            )
            out = layer_input * dropout_out
            out = self.decoder(out)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)

        return outputs

    def forward_step(
        self,
        layer: int,
        hidden: HiddenStates,
        input_: torch.FloatTensor,
        input_mask: torch.FloatTensor,
        time_mask: torch.FloatTensor,
    ) -> HiddenStates:
        """
        Do a single step for a single layer inside an LSTM. Intuitively, this can be seen as an upward-step inside the
        network, going from a lower layer to the one above.

        Parameters
        ----------
        layer: int
            Current layer number.
        hidden: HiddenStates
            Tuple of hidden and cell state from the previous time step.
        input_: torch.FloatTensor
            Input to the current layer: Either embedding if layer = 0 or hidden state from previous layer.
        input_mask: torch.FloatTensor
            Dropout masks applied to the input of the layer.
        time_mask: torch.FloatTensor
            Dropout masks applied on this layer between time steps.

        Returns
        -------
        HiddenStates
            New hidden and cell state for this layer.
        """
        hx, cx = hidden

        # Apply dropout masks
        hx = hx * time_mask
        input_ = input_ * input_mask

        # Forget gate
        f_g = torch.sigmoid(
            self.gates[layer]["if"](input_) + self.gates[layer]["hf"](hx)
        )

        # Input gate
        i_g = torch.sigmoid(
            self.gates[layer]["ii"](input_) + self.gates[layer]["hi"](hx)
        )

        # Output gate
        o_g = torch.sigmoid(
            self.gates[layer]["io"](input_) + self.gates[layer]["ho"](hx)
        )

        # Intermediate cell state
        c_tilde_g = torch.tanh(
            self.gates[layer]["ig"](input_) + self.gates[layer]["hg"](hx)
        )

        # New cell state
        cx = f_g * cx + i_g * c_tilde_g

        # New hidden state
        hx = o_g * torch.tanh(cx)

        return hx, cx

    def eval(self):
        # Manually turn off dropout
        self._dropout, self.dropout = self.dropout, 0
        self._input_dropout, self.input_dropout = self.input_dropout, 0
        super().eval()

    def train(self, *args):
        # Manually reinstate old dropout prob
        self.dropout = self._dropout
        self.input_dropout = self._input_dropout
        super().train(*args)


class LSTM(Model):
    """
    Model wrapper class for a LSTM.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "lstm", LSTMModule, model_params, train_params, model_dir, device
        )
