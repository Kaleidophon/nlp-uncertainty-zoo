"""
Implement the ST-tau LSTM by `Wang et al. (2021) <https://openreview.net/pdf?id=9EKHN1jOlA>`_.
"""

# STD
from typing import Optional, Tuple, Dict, Any, Type

# EXT
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

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
        vocab_size: int,
        output_size: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
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
        vocab_size: int
            Number of input vocabulary.
        output_size: int
            Number of classes.
        input_size: int
            Dimensionality of input to the first layer (embedding size).
        hidden_size: int
            Size of hidden units.
        num_layers: int
            Number of layers.
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
            vocab_size=vocab_size,
            output_size=output_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            is_sequence_classifier=is_sequence_classifier,
            device=device,
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
        vocab_size: int,
        output_size: int,
        input_size: int = 650,
        hidden_size: int = 650,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_centroids: int = 20,
        num_predictions: int = 10,
        is_sequence_classifier: bool = True,
        lr: float = 0.5,
        weight_decay: float = 0.001,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        scheduler_class: Optional[Type[scheduler._LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None,
        device: Device = "cpu",
        **model_params
    ):
        """
        Initialize a ST-tau LSTM model.

        Parameters
        ----------
        vocab_size: int
            Number of input vocabulary.
        output_size: int
            Number of classes.
        input_size: int
            Dimensionality of input to the first layer (embedding size). Default is 650.
        hidden_size: int
            Size of hidden units. Default is 650.
        num_layers: int
            Number of layers. Default is 2.
        dropout: float
            Dropout probability. Defailt is 0.2.
        num_centroids: int
            Number of states in the underlying finite-state automaton. Default is 20.
        num_predictions: int
            Number of predictions (forward passes) used to make predictions. Default is 10.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        lr: float
            Learning rate. Default is 0.5.
        weight_decay: float
            Weight decay term for optimizer. Default is 0.001.
        optimizer_class: Type[optim.Optimizer]
            Optimizer class. Default is Adam.
        scheduler_class: Optional[Type[scheduler._LRScheduler]]
            Learning rate scheduler class. Default is None.
        scheduler_kwargs: Optional[Dict[str, Any]]
            Keyword arguments for learning rate scheduler. Default is None.
        model_dir: Optional[str]
            Directory that model should be saved to.
        device: Device
            Device the model is located on.
        """
        super().__init__(
            "st_tau_lstm",
            STTauLSTMModule,
            vocab_size=vocab_size,
            output_size=output_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_centroids=num_centroids,
            num_predictions=num_predictions,
            is_sequence_classifier=is_sequence_classifier,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_class=optimizer_class,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            model_dir=model_dir,
            device=device,
            **model_params
        )
