"""
Define Bert modules used in this project and make them consistent with the other models in the repository.
"""

# EXT
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

# PROJECT
from nlp_uncertainty_zoo.models.model import Model, Module
from nlp_uncertainty_zoo.utils.custom_types import Device

# CONST
BERT_MODELS = {
    "english": "bert-base-uncased",
    "danish": "danbert-small-cased",
    "finnish": "bert-base-finnish-cased-v1",
    "swahili": "bert-base-multilingual-cased",
}

# TODO: Write Model subclass with BERT batching


class BERTModule(Module):
    """
    Define a BERT module that implements all the functions implemented in Module.
    """

    def __init__(
        self,
        language: str,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        assert (
            language in BERT_MODELS
        ), f"No Bert model has been specified for {language}!"

        self.bert = BertModel.from_pretrained(BERT_MODELS[language]).to(device)
        hidden_size = self.bert.config["hidden_size"]

        # Init custom pooler without tanh activations, copy Bert parameters
        self.custom_bert_pooler = nn.Linear(hidden_size, hidden_size)
        self.custom_bert_pooler.weight = self.bert.pooler.dense.weight
        self.custom_bert_pooler.bias = self.bert.pooler.dense.bias

        # Init layer norm
        if is_sequence_classifier:
            layer_norm_size = [hidden_size]
        else:
            layer_norm_size = [self.bert.config["max_position_embeddings"], hidden_size]

        self.layer_norm = nn.LayerNorm(layer_norm_size)

        super().__init__(
            num_layers=self.bert.config["num_hidden_layers"],
            vocab_size=self.bert.config["vocab_size"],
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            is_sequence_classifier=is_sequence_classifier,
            device=device,
        )

    def forward(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.
        Returns
        -------
        torch.FloatTensor
            Output predictions for input.
        """
        attention_mask = kwargs.get("attention_mask", args[0])
        return_dict = self.bert.forward(input_, attention_mask, return_dict=True)

        if self.is_sequence_classifier:
            cls_activations = return_dict["last_hidden_state"][:, 0, :]
            self.custom_bert_pooler(cls_activations)

            out = torch.tanh(self.custom_bert_pooler(cls_activations))
            out = self.layer_norm(out)

        else:
            activations = return_dict["hidden_states"]
            out = self.layer_norm(activations)

        return out

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
        return self.forward(input_, *args, **kwargs)

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        """
        Output a probability distribution over classes given an input. Results in a tensor of size batch_size x seq_len
        x output_size or batch_size x num_predictions x seq_len x output_size depending on the model type.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        logits = self.get_logits(input_, *args, **kwargs)
        probabilities = F.softmax(logits, dim=-1)

        return probabilities

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
        hidden = hidden[:, 0, :].unsqueeze(1)
        hidden = torch.tanh(self.custom_bert_pooler(hidden))

        return hidden
