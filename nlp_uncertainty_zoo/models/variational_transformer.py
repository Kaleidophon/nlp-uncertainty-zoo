"""
Implement the variational transformer, as presented by `(Xiao et al., 2021) <https://arxiv.org/pdf/2006.08344.pdf>`_.
"""

# STD
from typing import Dict, Any, Optional

# EXT
import torch
import torch.nn.functional as F

# PROJECT
from nlp_uncertainty_zoo.models.bert import BertModule
from nlp_uncertainty_zoo.models.model import Model, MultiPredictionMixin
from nlp_uncertainty_zoo.models.transformer import TransformerModule
from nlp_uncertainty_zoo.utils.custom_types import Device


class VariationalTransformerModule(TransformerModule, MultiPredictionMixin):
    """
    Implementation of Variational Transformer by `Xiao et al., (2021) <https://arxiv.org/pdf/2006.08344.pdf>`_.
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
        num_predictions: int,
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
        num_predictions: int
            Number of predictions with different dropout masks.
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
            input_dropout,
            dropout,
            num_heads,
            sequence_length,
            is_sequence_classifier,
            device,
        )
        MultiPredictionMixin.__init__(self, num_predictions)

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
            Number of predictions (number of forward passes).

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        if not num_predictions:
            num_predictions = self.num_predictions

        logits = torch.stack(
            [
                TransformerModule.get_logits(self, input_, *args, **kwargs)
                for _ in range(num_predictions)
            ],
            dim=1,
        )

        return logits

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        logits = self.get_logits(input_, *args, **kwargs)
        preds = F.softmax(logits, dim=-1).mean(dim=1)

        return preds

    def eval(self, *args):
        super().eval()

        for module in self._modules.values():
            if isinstance(module, torch.nn.Dropout):
                module.train()


class VariationalBertModule(BertModule, MultiPredictionMixin):
    """
    Implementation of Variational Transformer by `Xiao et al., (2021) <https://arxiv.org/pdf/2006.08344.pdf>`_ for BERT.
    """

    def __init__(
        self,
        bert_name: str,
        output_size: int,
        dropout: float,
        num_predictions: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize a transformer.

        Parameters
        ----------
        bert_name: str
            Name of the BERT to be used.
        dropout: float
            Dropout probability.
        num_predictions: int
            Number of predictions with different dropout masks.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            Device the model is located on.
        """
        self.num_predictions = num_predictions

        super().__init__(
            bert_name,
            output_size,
            is_sequence_classifier,
            device,
        )

        # Set dropout probability to argument
        for module in self.bert.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout

        MultiPredictionMixin.__init__(self, num_predictions)

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
            Number of predictions (number of forward passes).

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        if not num_predictions:
            num_predictions = self.num_predictions

        logits = torch.stack(
            [
                BertModule.get_logits(self, input_, *args, **kwargs)
                for _ in range(num_predictions)
            ],
            dim=1,
        )

        return logits

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        logits = self.get_logits(input_, *args, **kwargs)
        preds = F.softmax(logits, dim=-1).mean(dim=1)

        return preds

    def eval(self, *args):
        super().eval()

        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()


class VariationalTransformer(Model):
    """
    Module for the variational transformer.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "variational_transformer",
            VariationalTransformerModule,
            model_params,
            model_dir,
            device,
        )


class VariationalBert(Model):
    """
    Variational version of BERT.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        bert_name = model_params["bert_name"]
        super().__init__(
            f"variational-{bert_name}",
            VariationalBertModule,
            model_params,
            model_dir,
            device,
        )
