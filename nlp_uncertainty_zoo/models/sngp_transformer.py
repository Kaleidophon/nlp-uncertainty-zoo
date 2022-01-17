"""
Implementation of a Spectral-normalized Gaussian Process transformer as presented in
`Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_.
"""

# STD
import math
from typing import Tuple, Optional, Dict, Any

# EXT
from einops import rearrange
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# PROJECT
from nlp_uncertainty_zoo.datasets import DataSplit
from nlp_uncertainty_zoo.models.spectral import SpectralTransformerModule
from nlp_uncertainty_zoo.models.model import MultiPredictionMixin, Model
from nlp_uncertainty_zoo.utils.custom_types import Device


# TODO: Write version of this which accepts a pre-trained model that is to be fine-tuned


class SNGPModule(nn.Module):
    """
    Spectral-normalized Gaussian Process output layer, as presented in
    `Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_. Requires underlying model to contain residual
    connections in order to maintain bi-Lipschitz constraint.
    """

    def __init__(
        self,
        hidden_size: int,
        last_layer_size: int,
        output_size: int,
        ridge_factor: float,
        scaling_coefficient: float,
        beta_length_scale: float,
        gp_mean_field_factor: float,
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
        gp_mean_field_factor: float
            Multiplicative factor used in the mean-field approximation for the posterior mean of the softmax
            Gaussian process, based on `Lu et al. (2021) <https://arxiv.org/pdf/2006.07584.pdf>'_.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGP layer to come to the final prediction.
        device: Device
            Device the replication is performed on.
        """
        super().__init__()
        self.device = device

        self.hidden_size = hidden_size
        self.last_layer_size = last_layer_size
        self.output_size = output_size
        self.ridge_factor = ridge_factor
        self.scaling_coefficient = scaling_coefficient
        self.beta_length_scale = beta_length_scale
        self.gp_mean_field_factor = gp_mean_field_factor
        self.num_predictions = num_predictions

        # ### Init parameters

        # Random, frozen output layer
        self.output = nn.Linear(self.hidden_size, self.last_layer_size)
        # Change init of weights and biases following Liu et al. (2020)
        self.output.weight.data.normal_(0, 0.05)
        self.output.bias.data.uniform_(0, 2 * math.pi)

        # This layer is frozen right after init
        self.output.weight.requires_grad = False
        self.output.bias.requires_grad = False

        # Bundle all beta_k vectors into a matrix
        self.Beta = nn.Linear(last_layer_size, output_size)
        self.Beta.weight.data.normal_(0, beta_length_scale)
        self.Beta.bias.data = torch.zeros(output_size)

        # Initialize inverse of sigma hat, one matrix per class
        self.sigma_hat_inv = (
            torch.stack([torch.eye(last_layer_size) for _ in range(output_size)], dim=0)
            * self.ridge_factor
        ).to(device)
        self.sigma_hat = torch.zeros(
            output_size, last_layer_size, last_layer_size, device=device
        )
        self.inversed_sigma = False

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
        Phi = math.sqrt(2 / self.last_layer_size) * torch.cos(
            self.output(-x)
        )  # batch_size x last_layer_size
        # Logits: batch_size x last_layer_size @ last_layer_size x output_size -> batch_size x output_size
        post_mean = self.Beta(Phi)

        return post_mean, Phi

    def forward(
        self, x: torch.FloatTensor, update_sigma_hat_inv: bool = False
    ) -> torch.FloatTensor:
        """
        Forward pass for SNGP layer.

        Parameters
        ----------
        x: torch.FloatTensor
            Last hidden state of underlying model.
        update_sigma_hat_inv: bool
            Indicate whether the inverted sigma hat matrix should be updated (only during last training epoch).

        Returns
        -------
        torch.FloatTensor
            Logits for the current batch.
        """
        logits, Phi = self._get_features(x)

        if update_sigma_hat_inv:
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)  # batch_size x output_size
                Phi = Phi.unsqueeze(2)  # Make it batch_size x last_layer_size x 1

                # Vectorized version of eq. 9
                # P: probs * (1 - probs): batch_size x num_classes
                # PhiPhi: bos,bso->boo: Outer product along batch_dimension;
                # b: batch_size; o, p: last_layer_size; s: 1
                # Results in num_classes x last_layer_size x last_layer_size tensor to update sigma_hat_inv
                P = (probs * (1 - probs)).T
                PhiPhi = torch.einsum("bos,bsp->bop", Phi, torch.transpose(Phi, 1, 2))
                self.sigma_hat_inv *= self.scaling_coefficient
                self.sigma_hat += (1 - self.scaling_coefficient) * torch.einsum(
                    "kb,bop->kop", P, PhiPhi
                )

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

        assert (
            self.inversed_sigma
        ), "Sigma_hat matrix hasn't been inverted yet. Use invert_sigma_hat()."

        if num_predictions is None:
            num_predictions = self.num_predictions

        post_mean, Phi = self._get_features(x)

        # Compute posterior variance
        Phi = Phi.unsqueeze(2)  # Make it batch_size x last_layer_size x 1
        # Instance-wise outer-product: batch_size x last_layer_size x last_layer_size
        PhiPhi = torch.einsum("bos,bsp->bop", Phi, torch.transpose(Phi, 1, 2))
        post_var = torch.einsum("bop,pok->bk", PhiPhi, self.sigma_hat.T)

        out = 0
        for _ in range(num_predictions):
            # Now actually sample logits from posterior
            logits = torch.normal(post_mean, torch.sqrt(post_var + 1e-8))

            logits_scale = torch.sqrt(1 + post_var * self.gp_mean_field_factor)
            logits /= logits_scale

            # Adjust logits with mean field factor like done in implementation
            # here: https://github.com/google/uncertainty-baselines/blob/e854dfad5637cfae3561b67654c9c42ccabbe845/baseli
            # nes/clinc_intent/sngp.py#L408 and here: https://github.com/google/edward2/blob/89b59c1f3310266b0eaf175a7a
            # 28b048c727aaa2/edward2/tensorflow/layers/utils.py#L394
            # and originally based on this (https://arxiv.org/pdf/2006.07584.pdf) paper.

            preds = torch.softmax(logits, dim=-1)
            out += preds

        out /= num_predictions

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
        batch_size = x.shape[0]

        if num_predictions is None:
            num_predictions = self.num_predictions

        post_mean, Phi = self._get_features(x)

        # Compute posterior variance
        post_var = torch.zeros(
            Phi.shape[0], self.output_size, device=self.device
        )  # batch_size x output_size
        for k in range(self.output_size):
            post_var[:, k] = torch.diag(Phi @ self.sigma_hat[k, :, :] @ Phi.T)

        all_logits = torch.zeros(batch_size, num_predictions, self.output_size)
        for i in range(num_predictions):
            # Now actually sample logits from posterior
            logits = torch.normal(post_mean, torch.sqrt(post_var + 1e-8))
            logits_scale = torch.sqrt(1 + post_var * self.gp_mean_field_factor)
            logits /= logits_scale
            all_logits[:, i, :] = logits

        return all_logits

    def invert_sigma_hat(self) -> None:
        """
        Invert the sigma hat matrix. Because its one matrix per class, we invert one slice of a tensor here at a time.
        """
        for k in range(self.output_size):
            self.sigma_hat[k, :, :] = torch.inverse(self.sigma_hat_inv[k, :, :])

        self.inversed_sigma = True


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
        last_layer_size: int,
        output_size: int,
        input_dropout: float,
        dropout: float,
        num_heads: int,
        sequence_length: int,
        spectral_norm_upper_bound: float,
        ridge_factor: float,
        scaling_coefficient: float,
        beta_length_scale: float,
        gp_mean_field_factor: float,
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
        last_layer_size: int
            Size of last layer before output layer. Called D_L in the original paper.
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
        gp_mean_field_factor: float
            Multiplicative factor used in the mean-field approcimation for the posterior mean of the softmax
            Gaussian process, based on `Lu et al. (2021) <https://arxiv.org/pdf/2006.07584.pdf>'_.
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
            last_layer_size,
            output_size,
            ridge_factor,
            scaling_coefficient,
            beta_length_scale,
            gp_mean_field_factor,
            num_predictions,
            device,
        )
        self.layer_norm = nn.LayerNorm([input_size])

    def forward(self, input_: torch.LongTensor) -> torch.FloatTensor:
        out = self.get_hidden(input_)
        out = self.sngp_layer(out)

        return out

    def get_hidden(self, input_: torch.LongTensor) -> torch.FloatTensor:
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

        batch_size = input_.shape[0]
        sequence_length = input_.shape[1] if not self.is_sequence_classifier else 1
        out = self.get_hidden(input_)
        out = rearrange(out, "b t p -> (b t) p")
        out = self.sngp_layer.get_logits(out, num_predictions=num_predictions)
        out = rearrange(out, "(b t) n p -> b n t p", b=batch_size, t=sequence_length)

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
        data_split: DataSplit,
        verbose: bool,
        summary_writer: Optional[SummaryWriter] = None,
    ):
        # TODO: Refactor for new dataset usage
        self.module.sngp_layer.invert_sigma_hat()
