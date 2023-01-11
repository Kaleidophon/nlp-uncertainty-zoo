"""
Implement the variational transformer, as presented by `(Xiao et al., 2021) <https://arxiv.org/pdf/2006.08344.pdf>`_.
"""

# STD
from typing import Dict, Any, Optional, Type

# EXT
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import transformers

# PROJECT
from nlp_uncertainty_zoo.models.bert import BertModule, BertModel
from nlp_uncertainty_zoo.models.model import Model, MultiPredictionMixin
from nlp_uncertainty_zoo.models.transformer import TransformerModule
from nlp_uncertainty_zoo.utils.custom_types import Device


class VariationalTransformerModule(TransformerModule, MultiPredictionMixin):
    """
    Implementation of Variational Transformer by `Xiao et al., (2021) <https://arxiv.org/pdf/2006.08344.pdf>`_.
    """

    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
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
        Initialize a variational transformer.

        Parameters
        ----------
        vocab_size: int
            Vocabulary size.
        output_size: int
            Size of output of model.
        input_size: int
            Dimensionality of input to model.
        hidden_size: int
            Size of hidden representations.
        num_layers: int
            Number of model layers.
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
            vocab_size,
            output_size,
            input_size,
            hidden_size,
            num_layers,
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
        Initialize a variational BERT module.

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
        vocab_size: int,
        output_size: int,
        input_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 6,
        input_dropout: float = 0.2,
        dropout: float = 0.3,
        num_heads: int = 16,
        sequence_length: int = 128,
        num_predictions: int = 10,
        is_sequence_classifier: bool = True,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        scheduler_class: Type[scheduler._LRScheduler] = transformers.get_linear_schedule_with_warmup,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None,
        device: Device = "cpu",
        **model_params
    ):
        """
        Initialize a variational transformer model.

        Parameters
        ----------
        vocab_size: int
            Vocabulary size.
        output_size: int
            Size of output of model.
        input_size: int
            Dimensionality of input to model. Default is 650.
        hidden_size: int
            Size of hidden representations. Default is 650.
        num_layers: int
            Number of model layers. Default is 6.
        input_dropout: float
            Dropout on word embeddings. Default is 0.2.
        dropout: float
            Dropout rate. Default is 0.3.
        num_heads: int
            Number of self-attention heads per layer. Default is 16.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings. Default is 128.
        num_predictions: int
            Number of predictions with different dropout masks. Default is 10.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step. Default is True
        lr: float
            Learning rate. Default is 0.001.
        weight_decay: float
            Weight decay term for optimizer. Default is 0.01.
        optimizer_class: Type[optim.Optimizer]
            Optimizer class. Default is Adam.
        scheduler_class: Type[scheduler._LRScheduler]
            Learning rate scheduler class. Default is a triangular learning rate schedule.
        scheduler_kwargs: Optional[Dict[str, Any]]
            Keyword arguments for learning rate scheduler. If None, training length and warmup proportion will be set
            based on the arguments of fit(). Default is None.
        model_dir: Optional[str]
            Directory that model should be saved to.
        device: Device
            Device the model is located on.
        """
        super().__init__(
            model_name="variational_transformer",
            module_class=VariationalTransformerModule,
            vocab_size=vocab_size,
            output_size=output_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_dropout=input_dropout,
            dropout=dropout,
            num_heads=num_heads,
            sequence_length=sequence_length,
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


class VariationalBert(BertModel):
    """
    Variational version of BERT.
    """

    def __init__(
        self,
        bert_name: str,
        output_size: int,
        dropout: float = 0.3,
        num_predictions: int = 10,
        is_sequence_classifier: bool = True,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        scheduler_class: Optional[Type[scheduler._LRScheduler]] = transformers.get_linear_schedule_with_warmup,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None,
        device: Device = "cpu",
        **model_params
    ):
        """
        Initialize a variational BERT.

        Parameters
        ----------
        bert_name: str
            Name of the BERT to be used.
        output_size: int
            Size of output of model.
        dropout: float
            Dropout probability. Default is 0.3.
        num_predictions: int
            Number of predictions with different dropout masks. Default is 10.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        lr: float
            Learning rate. Default is 0.001.
        weight_decay: float
            Weight decay term for optimizer. Default is 0.01.
        optimizer_class: Type[optim.Optimizer]
            Optimizer class. Default is Adam.
        scheduler_class: Optional[Type[scheduler._LRScheduler]]
            Learning rate scheduler class. Default is a triangular learning rate schedule.
        scheduler_kwargs: Optional[Dict[str, Any]]
            Keyword arguments for learning rate scheduler. If None, training length and warmup proportion will be set
            based on the arguments of fit(). Default is None.
        model_dir: Optional[str]
            Directory that model should be saved to.
        device: Device
            Device the model is located on.
        """
        super().__init__(
            model_name=f"variational-{bert_name}",
            bert_name=bert_name,
            module_class=VariationalBertModule,
            output_size=output_size,
            dropout=dropout,
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
