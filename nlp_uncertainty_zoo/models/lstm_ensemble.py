"""
Implementation of an ensemble of LSTMs.
"""

# STD
from typing import Optional, Dict, Any

# EXT
from einops import rearrange
import torch
from torch import nn as nn

# PROJECT
from nlp_uncertainty_zoo.models.lstm import LSTMModule
from nlp_uncertainty_zoo.models.model import Module, MultiPredictionMixin, Model
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun


class LSTMEnsembleModule(Module, MultiPredictionMixin):
    """
    Implementation for an ensemble of LSTMs.
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        ensemble_size: int,
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
        ensemble_size: int
            Number of members in the ensemble.
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
        MultiPredictionMixin.__init__(self, ensemble_size)

        self.ensemble_size = ensemble_size
        self.ensemble_members = nn.ModuleList(
            [
                LSTMModule(
                    num_layers,
                    vocab_size,
                    input_size,
                    hidden_size,
                    output_size,
                    dropout,
                    is_sequence_classifier,
                    device,
                )
                for _ in range(ensemble_size)
            ]
        )

    def get_logits(
        self,
        input_: torch.LongTensor,
        *args,
        num_predictions: Optional[int] = None,
        **kwargs,
    ):

        if num_predictions is None:
            q, r = 1, 0
        else:
            q, r = divmod(num_predictions, len(self.ensemble_members))

        members = list(self.ensemble_members._modules.values())

        out = torch.stack(
            [member(input_) for member in q * members + members[:r]], dim=1
        )

        return out

    def forward(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        preds = self.get_logits(input_)

        return preds

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        preds = Module.predict(self, input_, *args, **kwargs)
        preds = preds.mean(dim=1)

        return preds

    def to(self, device: Device):
        """
        Move model to another device.

        Parameters
        ----------
        device: Device
            Device the model should be moved to.
        """
        for member in self.ensemble_members:
            member.to(device)

    @staticmethod
    def get_sequence_representation(hidden: torch.FloatTensor) -> torch.FloatTensor:
        """
        Create a sequence representation from an ensemble of LSTMs.

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


class LSTMEnsemble(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "lstm_ensemble",
            LSTMEnsembleModule,
            model_params,
            model_dir,
            device,
        )

    def get_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        wandb_run: Optional[WandBRun] = None,
        **kwargs,
    ) -> torch.Tensor:
        loss_function = nn.CrossEntropyLoss(
            ignore_index=-100, weight=self.loss_weights
        )  # Index that is used for non-masked tokens for MLM
        loss = torch.zeros(1, device=self.device)
        preds = self.module.forward(X, **kwargs)

        for n in range(self.module.ensemble_size):

            loss += loss_function(
                rearrange(preds[:, n], "b t p -> (b t) p"),
                rearrange(y, "b l -> (b l)")
                if not self.module.is_sequence_classifier
                else y,
            )

        loss /= self.module.ensemble_size

        return loss
