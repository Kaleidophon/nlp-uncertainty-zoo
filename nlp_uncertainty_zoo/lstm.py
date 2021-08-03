"""
Implement a simple vanilla LSTM.
"""

# STD
from typing import Dict, Any, Optional

# EXT
import torch
import torch.nn as nn
import torch.nn.functional as F

# PROJECT
from nlp_uncertainty_zoo.model import Model, Module
from nlp_uncertainty_zoo.types import Device, HiddenDict


class LSTMModule(Module):
    """
    Implementation of a LSTM for classification.
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        is_sequence_classifier: bool,
        device: Device,
    ):
        """
        Initialize an LSTM.

        Parameters
        ----------
        num_layers: int
            Number of layers.
        vocab_size: int
            Number of input vocabulary.
        input_size: int
            Dimensionality of input to the first layer (embedding size).
        hidden_size: int
            Size of hidden units.
        output_size: int
            Number of classes.
        dropout: float
            Dropout probability.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            Device the model should be moved to.
        """
        super().__init__(
            num_layers, vocab_size, input_size, hidden_size, output_size, device
        )

        # Initialize modules
        self.embeddings = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)
        self.is_sequence_classifier = is_sequence_classifier

        # Misc.
        self.last_hidden_states = None

    def forward(
        self,
        input_: torch.LongTensor,
        hidden_states: Optional[HiddenDict] = None,
    ) -> torch.FloatTensor:
        """
        The forward pass of the model.

        Parameters
        ----------
        input_: torch.LongTensor
            Current batch in the form of one-hot encodings.
        hidden_states: Optional[HiddenDict]
            Dictionary of hidden and cell states by layer to initialize the model with at the first time step. If None,
            they will be initialized with zero vectors or the ones stored under last_hidden_states if available.

        Returns
        -------
        torch.FloatTensor
            Tensor of unnormalized output distributions for the current batch.
        """
        batch_size, sequence_length = input_.shape

        # Initialize hidden activations if not given
        if hidden_states is None and self.last_hidden_states is None:
            hidden_states, cell_states = self.init_hidden_states(
                batch_size, self.device
            )
        # Detach hidden activations to limit gradient computations
        else:
            hidden_states, cell_states = (
                self.last_hidden_states if hidden_states is None else hidden_states
            )

        embeddings = self.embeddings(input_)
        out, (hidden_states, cell_states) = self.lstm(
            embeddings, (hidden_states, cell_states)
        )

        # Only use last hidden state for prediction
        if self.is_sequence_classifier:
            out = out[:, -1, :]

        out = self.dropout(out)
        out = self.output(out)

        self.last_hidden_states = hidden_states.detach(), cell_states.detach()

        return out

    def init_hidden_states(self, batch_size: int, device: Device):
        """
        Initialize all the hidden and cell states by zero vectors, for instance in the beginning of the training or
        after switching from test to training or vice versa.

        Parameters
        ----------
        batch_size: int
            Size of current batch.
        device: Device
            Device of the model.
        """
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )


class LSTM(Model):
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

        # Only for Zaremba et al. / Gal & Ghahramani replication, I know this isn't pretty
        if "init_weight" in train_params:
            init_weight = train_params["init_weight"]

            for layer_weights in self.module.lstm.all_weights:
                for param in layer_weights:
                    param.data.uniform_(-init_weight, init_weight)

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = super().predict(X)
        preds = F.softmax(out, dim=-1)

        return preds
