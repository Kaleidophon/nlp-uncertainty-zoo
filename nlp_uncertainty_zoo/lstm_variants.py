"""
Implement LSTM variants that only differ in the cell that they are using. Specifically, implement

- Bayes-by-backprop LSTM [1]
- ST-tau LSTM [2]

[1] https://arxiv.org/pdf/1704.02798.pdf
[2] https://openreview.net/pdf?id=9EKHN1jOlA
"""

# STD
from abc import ABC
from typing import Type, Dict, Any, Tuple, Optional, List

# EXT
from blitz.modules import BayesianLSTM as BlitzBayesianLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

# PROJECT
from nlp_uncertainty_zoo.lstm import LSTMModule
import nlp_uncertainty_zoo.metrics as metrics
from nlp_uncertainty_zoo.model import Model
from nlp_uncertainty_zoo.types import Device


class LayerWiseLSTM(nn.Module):
    """
    Model of a LSTM with a custom layer class.
    """

    def __init__(self, layers: List[nn.Module], dropout: float):
        """
        Initialize a LSTM with a custom layer class.

        Parameters
        ----------
        layers: List[nn.Module]
            List of layer objects.
        dropout: float
            Dropout probability for dropout applied between layers, except after the last layer.
        """
        super().__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_: torch.FloatTensor,
        hidden: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        hx, cx = hidden
        new_hx, new_cx = torch.zeros(hx.shape), torch.zeros(cx.shape)

        for l, layer in enumerate(self.layers):
            out, (new_hx[l, :], new_cx[l, :]) = layer(input_, (hx[l, :], cx[l, :]))
            input_ = self.dropout(out)

        return out, (new_hx, new_cx)


class BayesianLSTMModule(LSTMModule):
    """
    Implementation of a Bayes-by-backprop LSTM by Fortunato et al. (2017).
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        prior_sigma_1: float,
        prior_sigma_2: float,
        prior_pi: float,
        posterior_mu_init: float,
        posterior_rho_init: float,
        num_predictions: int,
        is_sequence_classifier: bool,
        device: Device,
    ):
        """
        Initialize a Bayesian LSTM.

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
        posterior_rho_init: float
            Posterior mean for the weight rho init.
        posterior_mu_init: float
            Posterior mean for the weight mu init.
        prior_pi: float
            Mixture weight of the prior.
        prior_sigma_1: float
            Prior sigma on the mixture prior distribution 1.
        prior_sigma_2: float
            Prior sigma on the mixture prior distribution 2.
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

        self.num_predictions = num_predictions
        self.lstm = LayerWiseLSTM(
            [
                BlitzBayesianLSTM(
                    in_features=input_size,
                    out_features=hidden_size,
                    prior_sigma_1=prior_sigma_1,
                    prior_sigma_2=prior_sigma_2,
                    prior_pi=prior_pi,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                )
                for _ in range(num_layers)
            ],
            dropout=dropout,
        )
        self.multi_prediction_uncertainty_metrics = {
            "variance": metrics.variance,
            "mutual_information": metrics.mutual_information,
        }
        self.default_uncertainty_metric = "variance"

    def get_logits(self, input_: torch.LongTensor) -> torch.FloatTensor:
        """
        Get the logits for an input. Results in a tensor of size batch_size x seq_len x output_size or batch_size x
        num_predictions x seq_len x output_size depending on the model type. Used to create inputs for the uncertainty
        metrics defined in nlp_uncertainty_zoo.metrics.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        out = [self.forward(input_) for _ in range(self.num_predictions)]
        out = torch.stack(out, dim=1)

        return out


class BayesianLSTM(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "bayesian_lstm",
            BayesianLSTMModule,
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


class CellWiseLSTM(nn.Module):
    """
    Model of a LSTM with a custom cell class.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        cells: List[nn.LSTMCell],
        device: Device,
    ):
        """

        Parameters
        ----------
        input_size: int
            Dimensionality of input to the first layer (embedding size).
        hidden_size: int
            Size of hidden units.
        dropout: float
            Dropout probability.
        cells: List[nn.Cell]
            List of cells, with one per layer.
        device: Device
            Device the model should be moved to.
        """
        super().__init__()
        self.cells = cells
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
        new_hx, new_cx = torch.zeros(hx.shape), torch.zeros(cx.shape)

        for t in range(sequence_length):
            input_t = input_[:, t, :]

            for layer, cell in enumerate(self.cells):
                new_hx[layer, :], new_cx[layer, :] = cell(
                    input_t, (hx[layer, :], cx[layer, :])
                )
                input_t = self.dropout(hx[layer, :])

            new_output[:, t] = hx[-1, :, :]

        return new_output, (new_hx, new_cx)


class STTauCell(nn.LSTMCell):
    """
    Implementation of a ST-tau LSTM cell, based on the implementation of Wang et al. (2021) [1].
    In contrast to the original implementation, the base cell is not a peephole-, but a normal LSTM cell.

    [1] https://github.com/nec-research/st_tau/blob/master/st_stau/st_tau.py
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_centroids: int, device: Device
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
        self.centroid_kernel = nn.Linear(self.hidden_size, num_centroids, bias=False)
        # Learnable temperature parameter for Gumbel softmax
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = super().forward(input, hx)

        logits = self.centroid_kernel(hidden)
        samples = F.gumbel_softmax(logits, tau=self.temperature)
        new_hidden = samples @ self.centroid_kernel.weight

        return new_hidden, cell


class STTauLSTMModule(LSTMModule):
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

        # Override LSTM
        self.lstm = CellWiseLSTM(
            input_size,
            hidden_size,
            dropout,
            cells,
            device,
        )

        self.num_predictions = num_predictions

    def get_logits(self, input_: torch.LongTensor) -> torch.FloatTensor:
        """
        Get the logits for an input. Results in a tensor of size batch_size x seq_len x output_size or batch_size x
        num_predictions x seq_len x output_size depending on the model type. Used to create inputs for the uncertainty
        metrics defined in nlp_uncertainty_zoo.metrics.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        out = [self.forward(input_) for _ in range(self.num_predictions)]
        out = torch.stack(out, dim=1)

        return out


class STTauLSTM(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "st_tau_lstm",
            STTauLSTMModule,
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
