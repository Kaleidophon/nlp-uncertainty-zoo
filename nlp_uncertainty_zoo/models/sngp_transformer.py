"""
Implementation of a Spectral-normalized Gaussian Process transformer as presented in
`Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_.
"""

# STD
import math
from typing import Tuple, Optional, Dict, Any
import warnings

# EXT
from einops import rearrange
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# PROJECT
from nlp_uncertainty_zoo.models.spectral import (
    SpectralTransformerModule,
    SpectralBertModule,
)
from nlp_uncertainty_zoo.models.model import MultiPredictionMixin, Model
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun


class SNGPModule(nn.Module):
    """
    Spectral-normalized Gaussian Process output layer, as presented in
    `Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_. Requires underlying model to contain residual
    connections in order to maintain bi-Lipschitz constraint.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        ridge_factor: float,
        scaling_coefficient: float,
        beta_length_scale: float,
        kernel_amplitude: float,
        num_predictions: int,
        device: Device,
        **build_params,
    ):
        """
        Initialize a SNGP output layer.

        Parameters
        ----------
        hidden_size: int
            Hidden size of last regular network layer.
        last_layer_size: int
            Size of last layer before output layer. Called D_L in the original paper.
        output_size: int
            Size of output layer, so number of classes.
        ridge_factor: float
            Factor that identity sigma hat matrices of the SNGP layer are multiplied by.
        scaling_coefficient: float
            Momentum factor that is used when updating the sigma hat matrix of the SNGP layer during the last training
            epoch.
        beta_length_scale: float
            Factor for the variance parameter of the normal distribution all beta parameters of the SNGP layer are
            initialized from.
        kernel_amplitude: float
            Kernel amplitude used when computing GP features.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGP layer to come to the final prediction.
        device: Device
            Device the replication is performed on.
        """
        super().__init__()
        self.device = device

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ridge_factor = ridge_factor
        self.scaling_coefficient = scaling_coefficient
        self.beta_length_scale = beta_length_scale
        self.kernel_amplitude = kernel_amplitude
        self.num_predictions = num_predictions

        # ### Init parameters

        # Random, frozen output layer
        self.output = nn.Linear(self.hidden_size, self.output_size)
        # Change init of weights and biases following Liu et al. (2020)
        self.output.weight.data.normal_(0, 0.05)
        self.output.bias.data.uniform_(0, 2 * math.pi)

        # This layer is frozen right after init
        self.output.weight.requires_grad = False
        self.output.bias.requires_grad = False

        # Bundle all beta_k vectors into a matrix
        self.Beta = nn.Linear(output_size, output_size)
        self.Beta.weight.data.normal_(0, beta_length_scale)
        self.Beta.bias.data = torch.zeros(output_size)

        # Initialize inverse of sigma hat, one matrix in total to save memory
        self.sigma_hat_inv = torch.eye(output_size, device=self.device) * self.beta_length_scale
        self.sigma_hat = torch.zeros(output_size, output_size, device=device)
        self.inversed_sigma = False

        # Multivariate normal distributions that beta columns are sampled from after inverting sigma hat
        self.beta_dists = [None for _ in range(output_size)]

    def _get_features(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get posterior mean / logits and Phi feature matrix given an input.

        Parameters
        ----------
        x: torch.FloatTensor
            Last hidden state of underlying model.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor]
            Tensors of posterior mean and Phi matrix.
        """
        Phi = math.sqrt(2 * self.kernel_amplitude ** 2 / self.output_size) * torch.cos(
            self.output(-x)
        )  # batch_size x last_layer_size
        # Logits: batch_size x last_layer_size @ last_layer_size x output_size -> batch_size x output_size
        post_mean = self.Beta(Phi)

        return post_mean, Phi

    def forward(
        self, x: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Forward pass for SNGP layer.

        Parameters
        ----------
        x: torch.FloatTensor
            Last hidden state of underlying model.

        Returns
        -------
        torch.FloatTensor
            Logits for the current batch.
        """
        logits, Phi = self._get_features(x)

        if self.training:
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)  # batch_size x seq_len x output_size
                max_probs = torch.max(probs, dim=-1)[0]  # batch_size x seq_len
                max_probs = max_probs * (1 - max_probs)

                Phi = Phi.unsqueeze(-1)  # Make it batch_size x last_layer_size x 1

                # Vectorized version of eq. 9
                # b: batch size
                # s: sequence length
                # o, p: output size
                # z: singleton dimension
                # Integrate multiplication with max_probs into einsum to avoid producing another 4D tensor
                PhiPhi = torch.einsum(
                    "bsoz,bszp->op",
                    Phi * max_probs.unsqueeze(-1).unsqueeze(-1),
                    torch.transpose(Phi, 2, 3)
                )
                self.sigma_hat_inv *= self.scaling_coefficient
                self.sigma_hat_inv += (1 - self.scaling_coefficient) * PhiPhi

        return logits

    def predict(self, x: torch.FloatTensor, num_predictions: Optional[int] = None):
        """
        Get predictions for the current batch.

        Parameters
        ----------
        x: torch.FloatTensor
            Last hidden state of underlying model.
        num_predictions: Optional[int]
            Number of predictions sampled from the GP in the SNGP layer to come to the final prediction. If None, number
            specified during initialization is used.

        Returns
        -------
        torch.FloatTensor
            Class probabilities for current batch.
        """

        logits = self.get_logits(x, num_predictions)

        out = F.softmax(logits, dim=-1).mean(dim=1)

        return out

    def get_logits(self, x: torch.FloatTensor, num_predictions: Optional[int] = None):
        """
        Get the logits for an input. Results in a tensor of size batch_size x num_predictions x seq_len x output_size
        depending on the model type.

        Parameters
        ----------
        x: torch.LongTensor
            Input to the model, containing all token IDs of the current batch.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGP layer to come to the final prediction. If None, number
            specified during initialization is used.

        Returns
        -------
        torch.FloatTensor
            Logits for the current batch.
        """

        if num_predictions is None:
            num_predictions = self.num_predictions

        # In case the Sigma matrix wasn't inverted yet, make sure that it is here.
        # Also set inversed_sigma to False again in case this is called during validation so that the matrix will be
        # updated over the training time again.
        if not self.inversed_sigma:
            self.invert_sigma_hat()
            self.inversed_sigma = False

        if num_predictions is None:
            num_predictions = self.num_predictions

        _, Phi = self._get_features(x)

        # Sample num_predictions Beta matrices
        beta_samples = torch.stack(
            [
                dist.rsample((num_predictions, ))
                for dist in self.beta_dists
            ],
            dim=-1
        )

        # Because we just stacked num_predictions samples for every column of the beta matrix, we now have to switch
        # the last two dimensions to obtain a num_predictions Beta matrices
        beta_samples = rearrange(beta_samples, "n c r -> n r c")

        # Now compute different logits using betas sampled from the Laplace posterior. This operation is a batch
        # instance-wise time step-wise multiplication of features with one Beta matrix per num_predictions.
        # b: batch
        # s: sequence length
        # k: number of classes
        # n: number of predictions
        logits = torch.einsum("bsk,nkk->bnsk", Phi, beta_samples)

        return logits

    def invert_sigma_hat(self) -> None:
        """
        Invert the sigma hat matrix.
        """
        try:
            self.sigma_hat = torch.inverse(self.sigma_hat_inv)

        except RuntimeError:
            warnings.warn(f"Matrix could not be inverted, compute pseudo-inverse instead.")
            self.sigma_hat = torch.linalg.pinv(self.sigma_hat_inv)

        self.inversed_sigma = True

        # Create multivariate normal distributions to sample columns from Beta matrix from. Updated every time
        # sigma_hat is updated since it sigma_hat is the covariance matrix used.
        self.beta_dists = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                self.Beta.weight.data[:, k], covariance_matrix=self.sigma_hat,
            )
            for k in range(self.output_size)
        ]


class SNGPTransformerModule(SpectralTransformerModule, MultiPredictionMixin):
    """
    Implementation of a spectral-normalized Gaussian Process transformer.
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
        spectral_norm_upper_bound: float,
        ridge_factor: float,
        scaling_coefficient: float,
        beta_length_scale: float,
        kernel_amplitude: float,
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
            Input dropout added to embeddings.
        dropout: float
            Dropout rate.
        num_heads: int
            Number of self-attention heads per layer.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings.
        spectral_norm_upper_bound: float
            Set a limit when weight matrices will be spectrally normalized if their eigenvalue surpasses it.
        ridge_factor: float
            Factor that identity sigma hat matrices of the SNGP layer are multiplied by.
        scaling_coefficient: float
            Momentum factor that is used when updating the sigma hat matrix of the SNGP layer during the last training
            epoch.
        beta_length_scale: float
            Factor for the variance parameter of the normal distribution all beta parameters of the SNGP layer are
            initialized from.
        kernel_amplitude: float
            Kernel amplitude used when computing GP features.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGP layer to come to the final prediction.
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
            spectral_norm_upper_bound,
            is_sequence_classifier,
            device,
        )
        MultiPredictionMixin.__init__(self, num_predictions)

        self.sngp_layer = SNGPModule(
            input_size,
            output_size,
            ridge_factor,
            scaling_coefficient,
            beta_length_scale,
            kernel_amplitude,
            num_predictions,
            device,
        )
        self.layer_norm = nn.LayerNorm([input_size])

    def forward(self, input_: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        out = self.get_hidden(input_)
        out = self.sngp_layer(out)

        return out

    def get_hidden(self, input_: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        word_embeddings = self.word_embeddings(input_)
        embeddings = self.pos_embeddings(word_embeddings)
        embeddings = self.input_dropout(embeddings)

        out = self.encoder(embeddings)

        if self.is_sequence_classifier:
            out = self.get_sequence_representation(out)

        out = self.output_dropout(out)
        out = self.layer_norm(out)

        return out

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
        num_predictions: int
            Number of predictions.

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        if not num_predictions:
            num_predictions = self.num_predictions

        out = self.get_hidden(input_)
        out = self.sngp_layer.get_logits(out, num_predictions=num_predictions)

        return out

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
        logits = self.get_logits(input_, *args, **kwargs)
        preds = F.softmax(logits, dim=-1).mean(dim=1)

        return preds


class SNGPTransformer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "sngp_transformer",
            SNGPTransformerModule,
            model_params,
            model_dir,
            device,
        )

    def _finetune(
        self,
        data_split: DataLoader,
        verbose: bool,
        wandb_run: Optional[WandBRun] = None,
    ):
        self.module.sngp_layer.invert_sigma_hat()


class SNGPBertModule(SpectralBertModule, MultiPredictionMixin):
    """
    Implementation of a spectral-normalized Gaussian Process transformer, based on a fine-tuned Bert.
    """

    def __init__(
        self,
        bert_name: str,
        output_size: int,
        spectral_norm_upper_bound: float,
        ridge_factor: float,
        scaling_coefficient: float,
        beta_length_scale: float,
        kernel_amplitude: float,
        num_predictions: int,
        is_sequence_classifier: bool,
        device: Device,
        **build_params,
    ):
        """
        Initialize a BERT with spectrally-normalized Gaussian Process output layer.

        Parameters
        ----------
        bert_name: str
            Name of the underlying BERT, as specified in HuggingFace transformers.
        output_size: int
            Number of classes.
        spectral_norm_upper_bound: float
            Set a limit when weight matrices will be spectrally normalized if their eigenvalue surpasses it.
        ridge_factor: float
            Factor that identity sigma hat matrices of the SNGP layer are multiplied by.
        scaling_coefficient: float
            Momentum factor that is used when updating the sigma hat matrix of the SNGP layer during the last training
            epoch.
        beta_length_scale: float
            Factor for the variance parameter of the normal distribution all beta parameters of the SNGP layer are
            initialized from.
        kernel_amplitude: float
            Kernel amplitude used when computing GP features.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGP layer to come to the final prediction.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        device: Device
            Device the model is located on.
        """
        super().__init__(
            bert_name,
            output_size,
            spectral_norm_upper_bound,
            is_sequence_classifier,
            device,
        )
        MultiPredictionMixin.__init__(self, num_predictions)

        hidden_size = self.bert.config.hidden_size
        self.sngp_layer = SNGPModule(
            hidden_size,
            output_size,
            ridge_factor,
            scaling_coefficient,
            beta_length_scale,
            kernel_amplitude,
            num_predictions,
            device,
        )
        self.layer_norm = nn.LayerNorm([hidden_size])

    def forward(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
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

        out = self.sngp_layer.forward(out)

        return out

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
        num_predictions: int
            Number of predictions.

        Returns
        -------
        torch.FloatTensor
            Logits for current input.
        """
        if not num_predictions:
            num_predictions = self.num_predictions

        out = self.get_hidden(input_, **kwargs)
        out = self.sngp_layer.get_logits(out, num_predictions=num_predictions)

        return out

    def predict(self, input_: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:
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

        probs = self.sngp_layer.predict(out)

        return probs


class SNGPBert(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        bert_name = model_params["bert_name"]
        super().__init__(
            f"sngp-{bert_name}",
            SNGPBertModule,
            model_params,
            model_dir,
            device,
        )
        self.weight_decay_beta = model_params["weight_decay_beta"]

    def _finetune(
        self,
        data_split: DataLoader,
        verbose: bool,
        wandb_run: Optional[WandBRun] = None,
    ):
        self.module.sngp_layer.invert_sigma_hat()

    def get_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        wandb_run: Optional[WandBRun] = None,
        **kwargs,
    ) -> torch.Tensor:
        loss = super().get_loss(X, y, wandb_run, **kwargs)

        # Compute weight decay for beta matrix separately
        loss += self.weight_decay_beta / 2 * torch.norm(self.module.sngp_layer.Beta.weight)

        return loss

