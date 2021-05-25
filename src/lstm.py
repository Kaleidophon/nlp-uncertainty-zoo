"""
Implement a vanilla LSTM model.
"""

# STD
import math
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
        embedding_dropout: float,
        layer_dropout: float,
        time_dropout: float,
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
        embedding_dropout: float
            Dropout on word embeddings. Dropout application corresponds to `Gal & Ghahramani (2016)
            <https://papers.nips.cc/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf>`_.
        layer_dropout: float
            Dropout rate between layers. Dropout application corresponds to `Gal & Ghahramani (2016)
            <https://papers.nips.cc/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf>`_.
        time_dropout: float
            Dropout rate between time steps. Dropout application corresponds to `Gal & Ghahramani (2016)
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
        self.embedding_dropout, self._embedding_dropout = (
            embedding_dropout,
            embedding_dropout,
        )
        self.layer_dropout, self._layer_dropout = layer_dropout, layer_dropout
        self.time_dropout, self._time_dropout = time_dropout, time_dropout
        self.last_hidden_states = None

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

            for name, gate in self.gates[layer].items():
                self.add_module(f"Layer {layer+1} / {name}", gate)

    def forward(
        self, input_: torch.LongTensor, hidden_states: Optional[HiddenDict] = None
    ) -> torch.Tensor:
        batch_size, sequence_length = input_.shape

        # Initialize hidden activations if not given
        if hidden_states is None and self.last_hidden_states is None:
            hidden_states = self.init_hidden_states(batch_size, self.device)
        # Detach hidden activations to limit gradient computations
        else:
            hidden_states = (
                self.last_hidden_states if hidden_states is None else hidden_states
            )

        # Sample all dropout masks used for this batch
        # Save some compute by initializing once
        mask_tensor = torch.ones(batch_size, self.hidden_size, device=self.device)
        dropout_masks_layer = (
            {  # Dropout mask applied to input of each layer, same across time steps
                layer: torch.bernoulli(mask_tensor * (1 - self.layer_dropout))
                for layer in range(self.num_layers)
            }
        )
        dropout_masks_time = (
            {  # Dropout mask applied between time steps, same for same layer
                layer: torch.bernoulli(mask_tensor * (1 - self.time_dropout))
                for layer in range(self.num_layers)
            }
        )
        new_hidden_states: HiddenDict = {}
        outputs = []

        # Sample types which are going to be zero'ed out
        types_to_drop = torch.randperm(self.vocab_size)[
            : math.floor(self.vocab_size * self.embedding_dropout)
        ].to(self.device)

        for t in range(sequence_length):

            embeddings = self.embeddings(input_[:, t])

            # TODO: Find a more elegant solution for this
            for i, in_ in enumerate(input_[:, t]):
                if in_ in types_to_drop:
                    embeddings[i, :] = 0

            layer_input = embeddings.squeeze(0)

            for layer in range(self.num_layers):
                new_hidden_state = self.forward_step(
                    layer,
                    hidden_states[layer],
                    layer_input,
                    dropout_masks_time[layer],
                )
                layer_input = new_hidden_state[
                    0
                ]  # New hidden state becomes input for next layer
                layer_input = (
                    layer_input * dropout_masks_layer[layer]
                )  # Apply dropout masks between layers
                new_hidden_states[layer] = new_hidden_state  # Store for next step

            out = self.decoder(layer_input)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        self._assign_last_hidden_states(new_hidden_states)

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

    def forward_step(
        self,
        layer: int,
        hidden_state: HiddenStates,
        input_: torch.FloatTensor,
        time_mask: torch.FloatTensor,
    ) -> HiddenStates:
        """
        Do a single step for a single layer inside an LSTM. Intuitively, this can be seen as an upward-step inside the
        network, going from a lower layer to the one above.

        Parameters
        ----------
        layer: int
            Current layer number.
        hidden_state: HiddenStates
            Tuple of hidden and cell state from the previous time step.
        input_: torch.FloatTensor
            Input to the current layer: Either embedding if layer = 0 or hidden state from previous layer.
        time_mask: torch.FloatTensor
            Dropout masks applied on this layer between time steps.

        Returns
        -------
        HiddenStates
            New hidden and cell state for this layer.
        """
        hx, cx = hidden_state

        # Apply dropout masks
        hx = hx * time_mask

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
        self._embedding_dropout, self.embedding_dropout = self.embedding_dropout, 0
        self._layer_dropout, self.layer_dropout = self.layer_dropout, 0
        self._time_dropout, self.time_dropout = self.time_dropout, 0

        # Reset hidden activations
        self.last_hidden_states = None

        super().eval()

    def train(self, *args):
        # Manually reinstate old dropout prob
        self.embedding_dropout = self.embedding_dropout
        self.layer_dropout = self._layer_dropout
        self.time_dropout = self._time_dropout

        # Reset hidden activations
        self.last_hidden_states = None

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
