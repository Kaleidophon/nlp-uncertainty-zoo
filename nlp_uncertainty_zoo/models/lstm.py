"""
Implement a simple vanilla LSTM.
"""

# STD
from typing import Dict, Any, Optional, List, Tuple

# EXT
import torch
import torch.nn as nn
import torch.nn.functional as F

# PROJECT
from nlp_uncertainty_zoo.models.model import Model, Module
from nlp_uncertainty_zoo.utils.custom_types import Device, HiddenDict


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
        **build_params,
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
            num_layers,
            vocab_size,
            input_size,
            hidden_size,
            output_size,
            is_sequence_classifier,
            device,
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
        **kwargs,
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
            out = self.get_sequence_representation(out)

        out = self.dropout(out)
        out = self.output(out)

        self.last_hidden_states = hidden_states.detach(), cell_states.detach()

        return out

    def get_sequence_representation(
        self, hidden: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Define how the representation for an entire sequence is extracted from a number of hidden states. This is
        relevant in sequence classification. For example, this could be the last hidden state for a unidirectional LSTM
        or the first hidden state for a transformer, adding a pooler layer.

        Parameters
        ----------
        hidden: torch.FloatTensor
            Hidden states of a model for a sequence.

        Returns
        -------
        torch.FloatTensor
            Representation for the current sequence.
        """
        return hidden[:, -1, :].unsqueeze(1)

    def get_logits(
        self, input_: torch.LongTensor, *args, **kwargs
    ) -> torch.FloatTensor:
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
        out = self.forward(input_)

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
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__("lstm", LSTMModule, model_params, model_dir, device)

        # Only for Zaremba et al. / Gal & Ghahramani replication, I know this isn't pretty
        if "init_weight" in model_params:
            init_weight = model_params["init_weight"]

            for layer_weights in self.module.lstm.all_weights:
                for param in layer_weights:
                    param.data.uniform_(-init_weight, init_weight)

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = super().predict(X)
        preds = F.softmax(out, dim=-1)

        return preds


class LayerWiseLSTM(nn.Module):
    """
    Model of a LSTM with a custom layer class.
    """

    def __init__(self, layers: List[nn.Module], dropout: float, device: Device):
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
        self.device = device

    def forward(
        self,
        input_: torch.FloatTensor,
        hidden: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        hx, cx = hidden
        new_hx, new_cx = torch.zeros(hx.shape, device=self.device), torch.zeros(cx.shape, device=self.device)

        for l, layer in enumerate(self.layers):
            input_ = self.dropout(input_)
            out, (new_hx[l, :], new_cx[l, :]) = layer(input_, (hx[l, :], cx[l, :]))
            input_ = out

        return out, (new_hx, new_cx)


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
        layer_hx = list(torch.split(hx, 1, dim=0))
        layer_cx = list(torch.split(cx, 1, dim=0))

        for t in range(sequence_length):
            input_t = input_[:, t, :]

            for layer, cell in enumerate(self.cells):
                input_t = self.dropout(input_t)
                input_t, (new_hx, new_cx) = cell(
                    input_t, (layer_hx[layer].squeeze(0), layer_cx[layer].squeeze(0))
                )
                layer_hx[layer] = new_hx.unsqueeze(0)
                layer_cx[layer] = new_cx.unsqueeze(0)

            new_output[:, t] = input_t

        hx, cx = torch.cat(layer_hx, dim=0), torch.cat(layer_cx, dim=0)

        return new_output, (hx, cx)
