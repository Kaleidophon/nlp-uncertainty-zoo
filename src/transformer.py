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
from src.module import Module, Model
from src.types import Device


# TODO: Use properly initialized pos embeddings


class Transformer(Model):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        num_heads: int,
        sequence_length: int,
        device: Device,
    ):
        super().__init__(
            num_layers, vocab_size, input_size, hidden_size, output_size, device
        )

        self.dropout = dropout
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.word_embeddings = nn.Embedding(vocab_size, input_size)
        self.pos_embeddings = nn.Embedding(sequence_length, input_size)

        self.output = nn.Linear(hidden_size, output_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=self.num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_: torch.LongTensor) -> torch.FloatTensor:
        word_embeddings = self.word_embeddings(input_)
        pos_embeddings = self.pos_embeddings(
            torch.arange(0, input_.shape[1]).repeat(input_.shape[0], 1)
        )
        embeddings = word_embeddings + pos_embeddings

        out = self.encoder(embeddings)
        out = self.output(out)

        return out


class TransformerModule(Module):
    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "transformer", Transformer, model_params, train_params, model_dir, device
        )
