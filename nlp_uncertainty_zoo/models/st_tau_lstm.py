"""
Implement the ST-tau LSTM by `Wang et al. (2021) <https://openreview.net/pdf?id=9EKHN1jOlA>`_.
"""

# STD
from typing import Optional, Tuple, Dict, Any

# EXT
import torch
from torch import nn as nn
from torch.nn import functional as F

# PROJECT
from nlp_uncertainty_zoo.models.lstm import CellWiseLSTM, LSTMModule
from nlp_uncertainty_zoo.models.model import MultiPredictionMixin, Model
from nlp_uncertainty_zoo.utils.custom_types import Device


class STTauCell(nn.LSTMCell):
    """
    Implementation of a ST-tau LSTM cell, based on the implementation of Wang et al. (2021) [1].
    In contrast to the original implementation, the base cell is not a peephole-, but a normal LSTM cell.

    [1] https://github.com/nec-research/st_tau/blob/master/st_stau/st_tau.py
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_centroids: int,
        device: Device,
        **build_params,
    ):
        """
        Initialize a ST-tau cell.

        Parameters
        ----------
        input_size: int
            Dimensionality of input to the first layer (embedding size).
        hidden_size: int
            Size of hidden units.
        num_centroids: int
            Number of states in the underlying finite-state automaton.
        device: Device
            Device the model should be moved to.
        """
        super().__init__(input_size, hidden_size, device=device)
        self.num_centroids = num_centroids
        # Transition matrix between states
        self.centroid_kernel = nn.Linear(self.hidden_size, num_centroids, bias=False, device=device)
        # Learnable temperature parameter for Gumbel softmax
        self.temperature = nn.Parameter(torch.ones(1, device=device), requires_grad=True)

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden, cell = super().forward(input, hx)

        logits = self.centroid_kernel(hidden)
        samples = F.gumbel_softmax(logits, tau=self.temperature)
        new_hidden = samples @ self.centroid_kernel.weight

        return hidden, (new_hidden, cell)


class STTauLSTMModule(LSTMModule, MultiPredictionMixin):
    """
    Implementation of a ST-tau LSTM by Wang et al. (2021).
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        num_centroids: int,
        num_predictions: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize a ST-tau LSTM.

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
        num_centroids: int
            Number of states in the underlying finite-state automaton.
        num_predictions: int
            Number of predictions (forward passes) used to make predictions.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            Device the model should be moved to.
        """
        super().__init__(
            num_layers,
            vocab_size,
            input_size,
            hidden_size,
            output_size,
            dropout,
            is_sequence_classifier,
            device,
        )
        MultiPredictionMixin.__init__(self, num_predictions)
        layer_sizes = [input_size] + [hidden_size] * num_layers
        cells = [
            STTauCell(
                input_size=in_size,
                hidden_size=out_size,
                device=device,
                num_centroids=num_centroids,
            ).to(device)
            for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        # No idea why this one is necessary but it is
        for i, cell in enumerate(cells):
            self.register_parameter(f"temperature_{i + 1}", cell.temperature)

            # Register original LSTM cell parameters
            for name, parameter in cell.named_parameters():
                self.register_parameter(f"{name.replace('.', '_')}_{i+1}", parameter)

        # Override LSTM
        self.lstm = CellWiseLSTM(
            input_size,
            hidden_size,
            dropout,
            cells,
            device,
        )

    def get_logits(
        self,
        input_: torch.LongTensor,
        *args,
        num_predictions: Optional[int] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Get the logits for an input. Results in a tensor of size batch_size x seq_len x output_size or batch_size x
        num_predictions x seq_len x output_size depending on the model type. Used to create inputs for the uncertainty
        metrics defined in nlp_uncertainty_zoo.metrics.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.
        num_predictions: Optional[int]
            Number of predictions (forward passes) used to make predictions.

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        if not num_predictions:
            num_predictions = self.num_predictions

        out = [self.forward(input_) for _ in range(num_predictions)]
        out = torch.stack(out, dim=1)

        return out

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        logits = self.get_logits(input_, *args, **kwargs)
        preds = F.softmax(logits, dim=-1).mean(dim=1)

        return preds


class STTauLSTM(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "st_tau_lstm",
            STTauLSTMModule,
            model_params,
            model_dir,
            device,
        )
