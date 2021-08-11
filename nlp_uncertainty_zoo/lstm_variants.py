"""
Implement LSTM variants that only differ in the cell that they are using. Specifically, implement

- Bayes-by-backprop LSTM (@TODO)
- ST-tau LSTM (@TODO)
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
    def __init__(self, layers: List[nn.Module], dropout: float):
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
    # TODO: Add doc
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
