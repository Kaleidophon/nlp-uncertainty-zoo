"""
Implement a vanilla transformer model.
"""

# STD
import math
from typing import Optional, Dict, Any

# EXT
import torch
import torch.nn as nn

# PROJECT
from nlp_uncertainty_zoo.models.model import Model, Module
from nlp_uncertainty_zoo.utils.custom_types import Device


class TransformerModule(Module):
    """
    Implementation of a transformer for classification.
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
        num_heads: int,
        sequence_length: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize a transformer.

        Parameters
        ----------
        num_layers: int
            Number of model layers.
        vocab_size: int
            Vocabulary size.
        input_size: int
            Dimensionality of input to model.
        hidden_size: int
            Size of hidden representations.
        output_size: int
            Size of output of model.
        input_dropout: float
            Dropout on word embeddings.
        dropout: float
            Dropout rate.
        num_heads: int
            Number of self-attention heads per layer.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            Device the model is located on.
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

        self.dropout = dropout
        self.input_dropout = nn.Dropout(input_dropout)
        self.output_dropout = nn.Dropout(self.dropout)
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.word_embeddings = nn.Embedding(vocab_size, input_size)
        self.pos_embeddings = PositionalEmbedding(sequence_length, input_size)

        self.pooler = nn.Linear(input_size, input_size)
        self.output = nn.Linear(input_size, output_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=self.num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
            device=self.device,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        hidden = self.get_hidden(input_)
        out = self.output_dropout(hidden)

        if self.is_sequence_classifier:
            out = self.get_sequence_representation(out)

        out = self.output(out)

        return out

    def get_hidden(
        self, input_: torch.LongTensor, *args, **kwargs
    ) -> torch.FloatTensor:
        word_embeddings = self.word_embeddings(input_)
        embeddings = self.pos_embeddings(word_embeddings)
        embeddings = self.input_dropout(embeddings)

        hidden = self.encoder(embeddings)

        return hidden

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
        return self.forward(input_)

    def get_sequence_representation(
        self, hidden: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Define how the representation for an entire sequence is extracted from a number of hidden states. This is
        relevant in sequence classification. In this case this is done by using the first hidden state and adding a
        pooler layer.

        Parameters
        ----------
        hidden: torch.FloatTensor
            Hidden states of a model for a sequence.

        Returns
        -------
        torch.FloatTensor
            Representation for the current sequence.
        """
        hidden = hidden[:, 0, :].unsqueeze(1)
        hidden = torch.tanh(self.pooler(hidden))

        return hidden


class Transformer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "transformer",
            TransformerModule,
            model_params,
            model_dir,
            device,
        )


class PositionalEmbedding(nn.Module):
    """
    Implementation of positional embeddings, shamelessly copied and adapted from the corresponding
    `PyTorch example <https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py>`_.
    """

    def __init__(self, sequence_length: int, input_size: int):
        """
        Initialize positional embeddings.

        Parameters
        ----------
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings.
        input_size: int
            Dimensionality of input to model.
        """
        super(PositionalEmbedding, self).__init__()

        positional_embeddings = torch.zeros(sequence_length, input_size)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_size, 2).float() * (-math.log(10000.0) / input_size)
        )
        positional_embeddings[:, 0::2] = torch.sin(position * div_term)
        positional_embeddings[:, 1::2] = torch.cos(position * div_term)
        positional_embeddings = positional_embeddings.unsqueeze(0)

        self.register_buffer("positional_embedding", positional_embeddings)

    def forward(self, input_: torch.FloatTensor) -> torch.FloatTensor:
        positional_embeddings = self.positional_embedding[:, : input_.shape[1], :]
        input_ = input_ + positional_embeddings

        return input_
