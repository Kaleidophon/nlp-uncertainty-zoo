"""
Implement a simple vanilla LSTM.
"""

# STD
from typing import Dict, Any, Optional, Tuple

# EXT
import torch
import torch.nn as nn
import torch.nn.functional as F

# PROJECT
from src.model import Model, Module
from src.types import Device


# TODO: Document


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
        device: Device,
    ):
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

        # Misc.
        self.last_hidden_states = None

    def forward(
        self,
        input_: torch.LongTensor,
        hidden_states: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
    ) -> torch.FloatTensor:
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
        out = self.dropout(out)
        out = self.output(out)

        self.last_hidden_states = hidden_states.detach(), cell_states.detach()

        return out

    def init_hidden_states(self, batch_size: int, device: Device):
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

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = super().predict(X)
        preds = F.softmax(out, dim=-1)

        return preds
