"""
Implement the Bayesian Bayes-by-backprop LSTM by `Fortunato et al. (2017) <https://arxiv.org/pdf/1704.02798.pdf>`_.
"""

# STD
from typing import Dict, Any, Optional

# EXT
import torch
import torch.nn.functional as F
from blitz.modules import BayesianLSTM as BlitzBayesianLSTM

# PROJECT
from nlp_uncertainty_zoo.models.lstm import LayerWiseLSTM, LSTMModule
from nlp_uncertainty_zoo.models.model import MultiPredictionMixin, Model
from nlp_uncertainty_zoo.utils.custom_types import Device


class BayesianLSTMModule(LSTMModule, MultiPredictionMixin):
    """
    Implementation of a Bayes-by-backprop LSTM by Fortunato et al. (2017).
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        prior_sigma_1: float,
        prior_sigma_2: float,
        prior_pi: float,
        posterior_mu_init: float,
        posterior_rho_init: float,
        num_predictions: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize a Bayesian LSTM.

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
        posterior_rho_init: float
            Posterior mean for the weight rho init.
        posterior_mu_init: float
            Posterior mean for the weight mu init.
        prior_pi: float
            Mixture weight of the prior.
        prior_sigma_1: float
            Prior sigma on the mixture prior distribution 1.
        prior_sigma_2: float
            Prior sigma on the mixture prior distribution 2.
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
            dropout,
            is_sequence_classifier,
            device,
        )
        MultiPredictionMixin.__init__(self, num_predictions)
        self.lstm = LayerWiseLSTM(
            [
                BlitzBayesianLSTM(
                    in_features=input_size,
                    out_features=hidden_size,
                    prior_sigma_1=prior_sigma_1,
                    prior_sigma_2=prior_sigma_2,
                    prior_pi=prior_pi,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                ).to(device)
                for _ in range(num_layers)
            ],
            dropout=dropout,
            device=device
        )

        for i, layer in enumerate(self.lstm.layers):

            # Register original Bayesian LSTM layer parameters
            for name, parameter in layer.named_parameters():
                self.register_parameter(f"{name.replace('.', '_')}_{i+1}", parameter)

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
            Number of predictions (forward passes) used to make predictions.

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        if not num_predictions:
            num_predictions = self.num_predictions

        out = [self.forward(input_) for _ in range(num_predictions)]
        out = torch.stack(out, dim=1)

        return out

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        logits = self.get_logits(input_, *args, **kwargs)
        preds = F.softmax(logits, dim=-1).mean(dim=1)

        return preds


class BayesianLSTM(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "bayesian_lstm",
            BayesianLSTMModule,
            model_params,
            model_dir,
            device,
        )
