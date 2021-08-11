"""
Implement LSTM variants that only differ in the cell that they are using. Specifically, implement

- Bayes-by-backprop LSTM (@TODO)
- ST-tau LSTM (@TODO)
"""

# STD
from abc import ABC
from typing import Type, Dict, Any, Tuple

# EXT
import torch
import torch.nn as nn

# PROJECT
from nlp_uncertainty_zoo.lstm import LSTM, LSTMModule
from nlp_uncertainty_zoo.types import Device


class CustomLSTMLogic(nn.Module, ABC):
    """
    Tries to imitate the interfaces of the torch LSTM module by providing the same input- and output structure,
    but using custom cells. Computations for sequences won't be as optimized as for nn.LSTM, but instead performed
    sequentially behind the scenes.

    Also, this class won't have quite as many functionalities as the original torch one, so it is batch_first and uni-
    directional in any case.

    The class will then be used by SequentialLSTMBase below, so that subclasses of SequentialLSTMBase only have to
    specify their custom cell type.
    """

    def __init__(
        self,
        num_layers: int,
        input_size: int,
        hidden_size: int,
        dropout: float,
        device: Device,
        lstm_cell_type: Type[nn.Module],
        lstm_cell_kwargs: Dict[str, Any],
    ):
        self.cells = [
            lstm_cell_type(input_size, hidden_size, **lstm_cell_kwargs).to(device)
            for _ in range(num_layers)
        ]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 0
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(
        self,
        input_: torch.FloatTensor,
        hidden: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        batch_size, sequence_length, _ = input_.shape
        # Define output variables
        new_output = torch.zeros(
            batch_size, sequence_length, self.hidden_size, device=self.device
        )

        # Unpack hidden
        hx, cx = hidden

        for t in range(sequence_length):
            input_t = input_[:, t, :]

            for layer, cell in enumerate(self.cells):
                hx[:, layer], cx[:, layer] = cell(input_t, (hx[:, layer], cx[:, layer]))
                input_t = self.dropout(hx[:, layer])

            new_output[:, t] = hx[:, -1]

        new_hx, new_cx = hx[-1, :, :], cx[-1, :, :]

        return new_output, (new_hx, new_cx)


class SequentialLSTMModule(LSTMModule, ABC):
    """
    Base class for LSTM-based models that do not operate on whole sequences, but time step by time step, because they
    implement a custom LSTM cell. Tries to otherwise stick to the LSTMModule structure.
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
        is_sequence_classifier: bool,
        device: Device,
        lstm_cell_type: Type[nn.Module],
        lstm_cell_kwargs: Dict[str, Any],
        **kwargs
    ):
        super().__init__(
            num_layers,
            vocab_size,
            input_size,
            hidden_size,
            output_size,
            input_dropout,
            dropout,
            is_sequence_classifier,
            device,
        )
        # Override definition of LSTM
        self.lstm = CustomLSTMLogic(
            num_layers,
            input_size,
            hidden_size,
            dropout,
            device,
            lstm_cell_type,
            lstm_cell_kwargs,
        )
