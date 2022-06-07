"""
Implementation of a variational LSTM, as presented by
`Gal & Ghrahramani (2016b) <https://arxiv.org/pdf/1512.05287.pdf>`.
"""

# STD
from typing import Optional, Dict, Any

# EXT
import torch
from torch import nn as nn
from torch.nn import functional as F

# PROJECT
from nlp_uncertainty_zoo.models.model import Module, MultiPredictionMixin, Model
from nlp_uncertainty_zoo.utils.custom_types import Device, HiddenDict


class VariationalDropout(nn.Module):
    """
    Variational Dropout module. In comparison to the default PyTorch module, this one only changes the dropout mask when
    sample() is called.
    """

    def __init__(self, dropout: float, input_dim: int, device: Device):
        super().__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.device = device
        self.mask = None

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.mask is None:
            raise ValueError("Dropout mask hasn't been sampled yet. Use .sample().")

        return x * self.mask

    def sample(self, batch_size: int):
        """
        Sample a new dropout mask for a batch of specified size.

        Parameters
        ----------
        batch_size: int
            Size of current batch.
        """
        self.mask = torch.bernoulli(
            torch.ones(batch_size, self.input_dim, device=self.device)
            * (1 - self.dropout)
        ) / (1 - self.dropout)


class VariationalLSTMModule(Module, MultiPredictionMixin):
    """
    Variational LSTM as described in `Gal & Ghrahramani (2016b) <https://arxiv.org/pdf/1512.05287.pdf>`, where the same
    dropout mask is being reused throughout a batch for connection of the same type.

    The only difference compared to the original implementation is that the embedding dropout works like normal dropout,
    not dropping out specific types. This was observed to yield minor improvement during experiments and simplified the
    implementation.
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
        num_predictions: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize a variational LSTM.

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
        embedding_dropout: float
            Dropout probability for the input embeddings.
        layer_dropout: float
            Dropout probability for hidden states between layers.
        time_dropout: float
            Dropout probability for hidden states between time steps.
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
            is_sequence_classifier,
            device,
        )
        MultiPredictionMixin.__init__(self, num_predictions)

        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTMCell(
                    in_size,
                    hidden_size,
                ).to(device)
                for in_size in [input_size] + [hidden_size] * (num_layers - 1)
            ]
        )

        self.embeddings = nn.Embedding(vocab_size, input_size)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.dropout_modules = {
            "embedding": [
                VariationalDropout(embedding_dropout, input_size, device)
            ],  # Use list here for consistency
            "layer": [
                VariationalDropout(layer_dropout, hidden_size, device)
                for _ in range(num_layers)
            ],
            "time": [
                VariationalDropout(time_dropout, hidden_size, device)
                for _ in range(num_layers)
            ],
        }

        self.last_hidden_states = None

    def forward(
        self,
        input_: torch.LongTensor,
        hidden_states: Optional[HiddenDict] = None,
        **kwargs,
    ):
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
        outputs = []

        # Initialize hidden activations if not given
        if hidden_states is None and self.last_hidden_states is None:
            hidden_states = self.init_hidden_states(batch_size, self.device)
        # Detach hidden activations to limit gradient computations
        else:
            hidden_states = (
                self.last_hidden_states if hidden_states is None else hidden_states
            )

        # Sample dropout masks used throughout this batch
        self.sample_masks(batch_size)

        for t in range(sequence_length):

            embeddings = self.embeddings(input_[:, t])
            layer_input = self.dropout_modules["embedding"][0](embeddings)
            # layer_input = self.dropout_modules["embedding"](embeddings, input_[:, t])

            for layer, cell in enumerate(self.lstm_layers):
                hx, cx = cell(
                    layer_input,  # Hidden state of last layer
                    (
                        self.dropout_modules["time"][layer](hidden_states[layer][0]),
                        hidden_states[layer][1],
                    ),  # Hidden and cell state state of last time step
                )
                layer_input = self.dropout_modules["layer"][layer](
                    hx
                )  # This becomes the input to the next layer
                hidden_states[layer] = (
                    hx,
                    cx,
                )  # This becomes the input for the next time step

            outputs.append(self.decoder(layer_input))

        outputs = torch.stack(outputs, dim=1)
        self._assign_last_hidden_states(hidden_states)

        # Only use last output
        if self.is_sequence_classifier:
            outputs = self.get_sequence_representation(outputs)

        return outputs

    def predict(
        self,
        input_: torch.LongTensor,
        hidden_states: Optional[HiddenDict] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Make a prediction for some input.

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
            Class probabilities for the current batch.
        """
        batch_size, seq_len = input_.shape

        # Make sure that the same hidden state from the last batch is used for all forward passes
        # Init hidden state - continue with hidden states from last batch
        hidden_states = self.last_hidden_states

        # This would e.g. happen when model is switched from train() to eval() - init hidden states with zeros
        if hidden_states is None:
            hidden_states = self.init_hidden_states(batch_size, self.device)

        logits = self.forward(input_, hidden_states)
        preds = F.softmax(logits, dim=-1)

        return preds

    def _assign_last_hidden_states(self, hidden: HiddenDict):
        """
        Assign hidden states at the end of a batch to an internal variable, detaching them from the computational graph.

        Parameters
        ----------
        hidden: HiddenDict
            Dictionary of hidden and cell states by layer.
        """
        self.last_hidden_states = {
            layer: (h[0].detach(), h[1].detach()) for layer, h in hidden.items()
        }

    def init_hidden_states(self, batch_size: int, device: Device) -> HiddenDict:
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
        hidden = {
            layer: (
                torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device),
            )
            for layer in range(self.num_layers)
        }

        return hidden

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

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        num_predictions: Optional[int]
            Number of predictions (forward passes) used to make predictions.
        """
        if not num_predictions:
            num_predictions = self.num_predictions

        out = [self.forward(input_) for _ in range(num_predictions)]
        out = torch.stack(out, dim=1)

        return out

    def sample_masks(self, batch_size: int):
        """
        Sample masks for the current batch.

        Parameters
        ----------
        batch_size: int
            Size of the current batch.
        """
        # Iterate over type of dropout modules ("layer", "time", "embedding")
        for dropout_modules in self.dropout_modules.values():
            # Iterate over all dropout modules of one type (across different layers)
            for layer_module in dropout_modules:
                layer_module.sample(batch_size)


class VariationalLSTM(Model):
    """
    Module for the variational LSTM.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "variational_lstm",
            VariationalLSTMModule,
            model_params,
            model_dir,
            device,
        )

        # Only for Gal & Ghahramani replication, I know this isn't pretty
        if "init_weight" in model_params:
            init_weight = model_params["init_weight"]

            for cell in self.module.lstm_layers:
                cell.weight_hh.data.uniform_(-init_weight, init_weight)
                cell.weight_ih.data.uniform_(-init_weight, init_weight)
                cell.bias_hh.data.uniform_(-init_weight, init_weight)
                cell.bias_ih.data.uniform_(-init_weight, init_weight)
