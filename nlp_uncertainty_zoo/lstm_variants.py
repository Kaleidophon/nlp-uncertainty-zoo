"""
Implement LSTM variants that only differ in the cell that they are using. Specifically, implement

- Bayes-by-backprop LSTM (@TODO)
- ST-tau LSTM (@TODO)
"""

# STD
from abc import ABC
from typing import Type

# EXT
import torch.nn as nn

# PROJECT
from nlp_uncertainty_zoo.lstm import LSTM, LSTMModule
from nlp_uncertainty_zoo.types import Device


class SequentialLSTMBase(LSTMModule, ABC):
    """
    Base class for LSTM-based models that do not operate on whole sequences, but time step by time step, because they
    implement a custom LSTM cell.
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
        sequence_length: int,
        is_sequence_classifier: bool,
        device: Device,
        lstm_cell_type: Type[nn.Module],
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
            sequence_length,
            is_sequence_classifier,
            device,
        )
        self.lstm = None
