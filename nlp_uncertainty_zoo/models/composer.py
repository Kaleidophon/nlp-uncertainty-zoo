"""
Import the composer model.
"""

# STD
import math
from typing import Optional, Tuple, Dict, Any

# EXT
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# PROJECT
from nlp_uncertainty_zoo.models.model import Model, Module
from nlp_uncertainty_zoo.models.transformer import PositionalEmbedding
from nlp_uncertainty_zoo.utils.custom_types import Device


# TODO: Add missing doc


class ComposerModule(Module):
    """
    Implementation of a composer for classification.
    """

    def __init__(
        self,
        num_layers: int,
        num_operations: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        sequence_length: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize a composer.

        Parameters
        ----------
        num_layers: int
            Number of model layers.
        num_operations: int
            Number of operations across layers.
        vocab_size: int
            Vocabulary size.
        hidden_size: int
            Size of hidden representations.
        output_size: int
            Size of output of model.
        dropout: float
            Dropout rate.
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

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.pos_embeddings = PositionalEmbedding(sequence_length, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.composer_layer = ComposerLayer(
            num_operations, hidden_size, sequence_length
        )
        self.pooler = nn.Linear(input_size, input_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.FloatTensor):
        x = self.word_embeddings(x)
        x = self.pos_embeddings(x)
        x = self.dropout(x)

        operator_queries = None

        for _ in range(self.num_layers):
            out, operator_queries = self.composer_layer(x, operator_queries)
            out = x + out
            x = out
            x = self.dropout(x)

        if self.is_sequence_classifier:
            x = self.get_sequence_representation(x)

        x = rearrange(x, "b s h -> (b s) h")
        out = self.output_layer(x)
        out = rearrange(out, "(b s) o -> b s o", s=self.sequence_length)

        return out

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
        hidden = hidden[:, 0, :]
        hidden = torch.tanh(self.pooler(hidden)).unsqueeze(1)

        return hidden


class ComposerLayer(nn.Module):
    def __init__(self, num_operations: int, hidden_size: int, sequence_length: int):
        super().__init__()

        self.num_operations = num_operations
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        # Define general parts
        self.values_hidden = nn.Linear(hidden_size, hidden_size)

        identity_operator = nn.Parameter(torch.ones(sequence_length, 1))
        identity_operator.requires_grad = False
        self.operators = [identity_operator]  # Add identity function

        for _ in range(num_operations - 1):
            self.operators.append(
                nn.Parameter(torch.randn(sequence_length, 1))
            )  # Add learned functions

        # Define parts of operator mechanism
        self.query_hidden = nn.Linear(
            hidden_size, hidden_size
        )  # Learned matrix to get query vectors for hidden reprs.

        # Define parts of relevance mechanism
        self.query_operators = nn.Linear(sequence_length, hidden_size)
        self.keys_hidden = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, x: torch.FloatTensor, operator_queries: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        if operator_queries is None:
            operator_queries = torch.stack(
                [
                    self.query_operators(operator.data.T).T.squeeze()
                    for operator in self.operators
                ]
            )  # num_operators x hidden_size

        # Compute representations of input
        input_values = self.values_hidden(x)  # batch_size x seq_len x hidden_size
        input_keys = self.keys_hidden(x)  # batch_size x seq_len x hidden_size

        # Computations of relevance mechanism
        relevance_logits = torch.einsum(
            "bsh,ho->bso", input_keys, operator_queries.T
        ) / math.sqrt(
            self.hidden_size
        )  # batch_size x seq_len x num_operators
        relevance_logits = rearrange(relevance_logits, "b s o -> b o s")
        relevance_weights = F.softmax(relevance_logits, dim=-1)

        # Grab the attention weights corresponding to the current operators, compute its result
        operator_outputs = []

        for o, operator in enumerate(self.operators):
            operator_relevance_weights = relevance_weights[
                :, o, :
            ]  # batch_size x seq_len
            weighted_input = torch.einsum(
                "bsh,bs->bsh", input_values, operator_relevance_weights
            )  # batch_size x seq_len x hidden
            operator_output = torch.einsum(
                "bsh,s->bh", weighted_input, operator.squeeze()
            )  # batch_size x hidden
            operator_outputs.append(operator_output)

        operator_outputs = torch.stack(
            operator_outputs, dim=1
        )  # batch_size x num_operators x hidden_size

        # Computations of operator mechanism
        operator_logits = torch.einsum(
            "bsh,ho->bso", input_keys, operator_queries.T
        ) / math.sqrt(
            self.hidden_size
        )  # batch_size x seq_len x num_operators
        operator_weights = F.softmax(operator_logits, dim=-1)
        # Now, for every time step, weight the output of every operator to get final output
        output = torch.einsum(
            "bso,boh->bsh", operator_weights, operator_outputs
        )  # batch_size x seq_len x hidden_size

        return output, operator_queries


class Composer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "composer",
            ComposerModule,
            model_params,
            model_dir,
            device,
        )
