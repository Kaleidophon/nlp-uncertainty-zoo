"""
Define BERT modules used in this project and make them consistent with the other models in the repository.
"""

from typing import Type, Optional, Dict, Any

# EXT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from torch.utils.data import DataLoader
from transformers import BertModel as HFBertModel  # Rename to avoid collision

# PROJECT
from nlp_uncertainty_zoo.models.model import Module, Model
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun


class BertModule(Module):
    """
    Define a BERT module that implements all the functions implemented in Module.
    """

    def __init__(
        self,
        bert_name: str,
        output_size: int,
        is_sequence_classifier: bool,
        bert_class: Type[HFBertModel],
        device: Device,
        **build_params,
    ):
        """
        Initialize a BERT module.

        Parameters
        ----------
        bert_name: str
            Name of the underlying BERT, as specified in HuggingFace transformers.
        output_size: int
            Number of classes.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        bert_class: Type[HFBertModel]
            Type of BERT to be used. Default is BertModel from the Huggingface transformers package.
        device: Device
            Device the model should be moved to.
        """

        bert = bert_class.from_pretrained(bert_name).to(device)
        hidden_size = bert.config.hidden_size

        super().__init__(
            num_layers=bert.config.num_hidden_layers,
            vocab_size=bert.config.vocab_size,
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            is_sequence_classifier=is_sequence_classifier,
            device=device,
        )

        self.bert = bert
        self.output_size = output_size
        self.sequence_length = bert.config.max_length

        self.layer_norm = nn.LayerNorm([hidden_size])
        self.output = nn.Linear(hidden_size, output_size)

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
        attention_mask = kwargs["attention_mask"]
        return_dict = self.bert.forward(input_, attention_mask, return_dict=True)

        if self.is_sequence_classifier:
            cls_activations = return_dict["last_hidden_state"][:, 0, :]
            out = torch.tanh(self.bert.pooler.dense(cls_activations))
            out = self.layer_norm(out)
            out = out.unsqueeze(1)

        else:
            activations = return_dict["last_hidden_state"]
            out = self.layer_norm(activations)

        out = self.output(out)

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

    def get_sequence_representation_from_hidden(
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
        hidden = torch.tanh(self.bert.pooler.dense(hidden))

        return hidden

    def get_hidden_representation(
        self, input_: torch.LongTensor, *args, **kwargs
    ) -> torch.FloatTensor:
        attention_mask = kwargs["attention_mask"]
        return_dict = self.bert.forward(input_, attention_mask, return_dict=True)

        if self.is_sequence_classifier:
            activations = return_dict["last_hidden_state"][:, 0, :].unsqueeze(1)

        else:
            activations = return_dict["last_hidden_state"]

        return activations


class BertModel(Model):
    """
    Define a BERT model. The only purpose this serves it to provide a warmup_proportion for fit(). Since the number of
    training steps is only defined in fit(), it means we can only define the scheduler in that method.
    """
    def __init__(
        self,
        model_name: str,
        module_class: type,
        bert_name: str,
        output_size: int,
        is_sequence_classifier: bool,
        lr: float,
        weight_decay: float,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        scheduler_class: Optional[Type[scheduler._LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        bert_class: Type[HFBertModel] = HFBertModel,
        model_dir: Optional[str] = None,
        device: Device = "cpu",
        **model_params,
    ):
        """
        Initialize a BERT model.

        Parameters
        ----------
        model_name: str
            Name of the model.
        module_class: type
            Class of the model that is being wrapped.
        bert_name: str
            Name of the underlying BERT, as specified in HuggingFace transformers.
        output_size: int
            Number of classes.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        lr: float
            Learning rate. Default is 0.4931.
        weight_decay: float
            Weight decay term for optimizer. Default is 0.001357.
        optimizer_class: Type[optim.Optimizer]
            Optimizer class. Default is Adam.
        scheduler_class: Optional[Type[scheduler._LRScheduler]]
            Learning rate scheduler class. Default is None.
        scheduler_kwargs: Optional[Dict[str, Any]]
            Keyword arguments for learning rate scheduler. Default is None.
        bert_class: Type[HFBertModel]
            Type of BERT to be used. Default is BertModel from the Huggingface transformers package.
        model_dir: Optional[str]
            Directory that model should be saved to.
        device: Device
            Device the model is located on.
        """

        super().__init__(
            model_name=model_name,
            module_class=module_class,
            bert_name=bert_name,
            output_size=output_size,
            is_sequence_classifier=is_sequence_classifier,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_class=optimizer_class,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            model_dir=model_dir,
            bert_class=bert_class,
            device=device,
            **model_params,
        )

    def fit(
        self,
        train_split: DataLoader,
        num_training_steps: int,
        warmup_proportion: float = 0.1,
        valid_split: Optional[DataLoader] = None,
        weight_loss: bool = False,
        grad_clip: float = 10,
        validation_interval: Optional[int] = None,
        early_stopping_pat: int = np.inf,
        early_stopping: bool = False,
        verbose: bool = True,
        wandb_run: Optional[WandBRun] = None,
        **training_kwargs
    ):
        """
        Fit the model to training data.

        Parameters
        ----------
        train_split: DataLoader
            Dataset the model is being trained on.
        num_training_steps: int
            Number of training steps until completion.
        warmup_proportion: float
            Percentage of warmup steps for triangular learning rate schedule. Default is 0.1.
        valid_split: Optional[DataLoader]
            Validation set the model is being evaluated on if given.
        verbose: bool
            Whether to display information about current loss.
        weight_loss: bool
            Weight classes in loss function. Default is False.
        grad_clip: float
            Parameter grad norm value before it will be clipped. Default is 10.
        validation_interval: Optional[int]
            Interval of training steps between validations on the validation set. If None, the model is evaluated after
            each pass through the training data.
        early_stopping_pat: int
            Patience in number of training steps before early stopping kicks in. Default is np.inf.
        early_stopping: bool
            Whether early stopping should be used. Default is False.
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        assert 0 <= warmup_proportion <= 1, f"warmup_proportion should be in [0, 1], {warmup_proportion} found."

        if self.model_params.get("scheduler_class", None) is not None:
            scheduler_class = self.model_params["scheduler_class"]
            scheduler_kwargs = self.model_params.get("scheduler_kwargs", {})
            scheduler_kwargs = {
                # Warmup prob: 0.1
                "num_warmup_steps": int(num_training_steps * warmup_proportion),
                "num_training_steps": num_training_steps,
                **scheduler_kwargs
            }
            self.scheduler = scheduler_class(
                self.optimizer, **scheduler_kwargs
            )

        # Now call rest of function
        super().fit(
            train_split=train_split,
            num_training_steps=num_training_steps,
            valid_split=valid_split,
            weight_loss=weight_loss,
            grad_clip=grad_clip,
            validation_interval=validation_interval,
            early_stopping_pat=early_stopping_pat,
            early_stopping=early_stopping,
            verbose=verbose,
            wandb_run=wandb_run,
            **training_kwargs
        )
