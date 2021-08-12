"""
Implement transformer models that use spectral normalization to meet the bi-Lipschitz condition. More precisely,
this module implements a mixin enabling spectral normalization and, inheriting from that, the following two models:

* Spectral-normalized Gaussian Process (SNGP) Transformer (`Liu et al., 2020 <https://arxiv.org/pdf/2006.10108.pdf>`)
* Deterministic Uncertainty Estimation (DUE) Transformer
(`Van Amersfoort et al., 2021 <https://arxiv.org/pdf/2102.11409.pdf>`)
* Deep Deterministic Uncertainty (DDU) Transformer (`Mukhoti et al., 2021 <https://arxiv.org/pdf/2102.11582.pdf>`)
"""

# STD
import math
from typing import Tuple

# EXT
from due.dkl import GP, _get_initial_inducing_points, _get_initial_lengthscale
from due.layers.spectral_norm_fc import spectral_norm_fc
from einops import rearrange
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional

# PROJECT
from nlp_uncertainty_zoo.datasets import DataSplit, TextDataset
from nlp_uncertainty_zoo.transformer import TransformerModule
from nlp_uncertainty_zoo.model import Model, MultiPredictionMixin
from nlp_uncertainty_zoo.types import Device


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
            Multiplicative factor used in the mean-field approcimation for the posterior mean of the softmax
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
        batch_size, seq_len, _ = x.shape

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


class SpectralTransformerModule(TransformerModule):
    """
    Implementation of a spectral-normalized transformer. Used as a base for models like SNGP, DUE and DDU.
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
        is_sequence_classifier: bool,
        device: Device,
    ):
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

        # Add spectral normalization
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                setattr(
                    self,
                    module_name,
                    spectral_norm_fc(module, coeff=spectral_norm_upper_bound),
                )


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
        out = self.get_hidden(input_)
        out = self.sngp_layer.get_logits(out, num_predictions=self.num_predictions)

        return out


class SNGPTransformer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "sngp_transformer",
            SNGPTransformerModule,
            model_params,
            train_params,
            model_dir,
            device,
        )

    def predict(
        self, X: torch.Tensor, *args, num_predictions: Optional[int] = None
    ) -> torch.Tensor:
        """
        Make a prediction for some input.

        Parameters
        ----------
        X: torch.Tensor
            Input data points.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGP layer to come to the final prediction.

        Returns
        -------
        torch.Tensor
            Predictions.
        """
        if num_predictions is None:
            num_predictions = self.module.num_predictions

        X = X.to(self.device)

        word_embeddings = self.module.word_embeddings(X)
        embeddings = self.module.pos_embeddings(word_embeddings)
        embeddings = self.module.input_dropout(embeddings)

        out = self.module.encoder(embeddings)
        out = self.module.output_dropout(out)
        out = self.module.layer_norm(out)
        out = self.module.sngp_layer.predict(out, num_predictions=num_predictions)

        return out


class DUETransformerModule(SpectralTransformerModule, MultiPredictionMixin):
    """
    Implementation of Deterministic Uncertainty Estimation (DUE) Transformer by
    `Van Amersfoort et al., 2021 <https://arxiv.org/pdf/2102.11409.pdf>`.
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
        num_inducing_samples: int,
        num_inducing_points: int,
        spectral_norm_upper_bound: float,
        kernel_type: str,
        is_sequence_classifier: bool,
        device: Device,
    ):

        """
        Initialize a DDU transformer.

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
        kernel_type: str
            Define the type of kernel used. Can be one of {'RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ'}.
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

        self.num_inducing_samples = num_inducing_samples
        self.num_inducing_points = num_inducing_points
        self.spectral_norm_upper_bound = spectral_norm_upper_bound
        self.kernel_type = kernel_type
        self.layer_norm = nn.LayerNorm([input_size])

        self.gp = None
        self.likelihood = None
        self.loss_function = None

    def init_gp(self, train_data: DataSplit, num_instances: int = 1000):
        """
        Initialize the Gaussian Process layer together with the likelihood and loss function.

        Parameters
        ----------
        train_data: DataSplit
            Training split.
        num_instances: int
            Number of instances being sampled to initialize the GP inducing points and length scale.
        """
        # Compute how many batches need to be sampled to initialize the inducing points when using batches of
        # batch_size and length sequence_length
        batch_size = train_data[0][0].shape[0]
        num_batches = math.ceil(num_instances / (batch_size * self.sequence_length))

        # Essentially do the same as in due.dkl.initial_values_for_GP, but with a sequential dataset
        sampled_batch_idx = torch.randperm(len(train_data))[:num_batches]

        # Extract feature representations for sampled batches
        batch_representations = []

        with torch.no_grad():
            for batch_idx in sampled_batch_idx:
                X = train_data[batch_idx][0].to(self.device)
                batch_representations.append(self.get_hidden(X))

        representations = torch.cat(batch_representations, dim=0)
        representations = rearrange(representations, "b s h -> (b s) h")

        initial_inducing_points = _get_initial_inducing_points(
            representations.cpu().numpy(), self.num_inducing_points
        )
        initial_length_scale = _get_initial_lengthscale(representations)

        self.gp = GP(
            num_outputs=self.output_size,
            initial_lengthscale=initial_length_scale,
            initial_inducing_points=initial_inducing_points,
            kernel=self.kernel_type,
        ).to(self.device)
        self.likelihood = SoftmaxLikelihood(
            num_features=self.input_size,
            num_classes=self.output_size,
            mixing_weights=False,
        ).to(self.device)
        self.loss_function = VariationalELBO(
            self.likelihood, self.gp, num_data=len(train_data)
        ).to(self.device)

    def forward(self, input_: torch.LongTensor):
        out = self.get_hidden(input_)

        if self.is_sequence_classifier:
            out = self.get_sequence_representation(out)

        out = self.layer_norm(out)
        out = rearrange(out, "b s h -> (b s) h").float()
        mvn = self.gp(out)

        return mvn

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
        batch_size = input_.shape[0]

        mvn = self.forward(input_)
        predictions = mvn.sample(sample_shape=torch.Size((self.num_predictions,)))
        predictions = rearrange(
            predictions, "n (b s) o  -> b n s o", b=batch_size, n=self.num_predictions
        )

        return predictions


class DUETransformer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "ddu_transformer",
            DUETransformerModule,
            model_params,
            train_params,
            model_dir,
            device,
        )

    def fit(
        self,
        dataset: TextDataset,
        validate: bool = True,
        verbose: bool = True,
        summary_writer: Optional[SummaryWriter] = None,
    ):
        """
        Fit the model to training data.

        Parameters
        ----------
        dataset: TextDataset
            Dataset the model is being trained on.
        validate: bool
            Indicate whether model should also be evaluated on the validation set.
        verbose: bool
            Whether to display information about current loss.
        summary_writer: Optional[SummaryWriter]
            Summary writer to track training statistics. Training and validation loss (if applicable) are tracked by
            default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        # Retrieve inducing points and length scale from training set to initialize the GP
        self.module.init_gp(dataset.train)

        return super().fit(dataset, validate, verbose, summary_writer)

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        with torch.no_grad():
            out = self.module(X)
            out = self.module.likelihood(out)
            out = out.logits.mean(dim=0)

        return out

    def get_loss(
        self,
        n_batch: int,
        X: torch.Tensor,
        y: torch.Tensor,
        summary_writer: Optional[SummaryWriter] = None,
    ) -> torch.Tensor:
        """
        Get loss for a single batch. This uses the Variational ELBO instead of a cross-entropy loss.

        Parameters
        ----------
        n_batch: int
            Number of the current batch.
        X: torch.Tensor
            Batch input.
        y: torch.Tensor
            Batch labels.
        summary_writer: SummaryWriter
            Summary writer to track training statistics.

        Returns
        -------
        torch.Tensor
            Batch loss.
        """
        # TODO: This needs to be adapted for language modelling
        preds = self.module(X)
        loss = -self.module.loss_function(preds, y)

        return loss


class DDUTransformerModule(SpectralTransformerModule):
    """
    Implementation of the Deep Deterministic Uncertainty (DDU) Transformer by
    `Mukhoti et al., 2021 <https://arxiv.org/pdf/2102.11582.pdf>`.
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
        is_sequence_classifier: bool,
        device: Device,
    ):
        """
        Initialize a DDU transformer.

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

        # Parameters for Gaussian Discriminant Analysis
        self.mu = torch.zeros(output_size, input_size)
        self.Sigma = torch.stack(
            [torch.eye(input_size, input_size) for _ in range(self.output_size)]
        )
        self.determinants = torch.zeros(output_size)

    def gmm_fit(self, data_split: DataSplit) -> None:
        """
        Fit a Gaussian mixture model on the feature representations of the trained model.

        Parameters
        ----------
        data_split: DataSplit
            Data split used for fitting, usually the training or validation split.
        """
        with torch.no_grad():
            hiddens, labels = [], []

            for i, (X, y) in enumerate(data_split):

                hidden = self.get_hidden(X)
                hidden = (
                    self.get_sequence_representation(hidden).squeeze(1)
                    if self.is_sequence_classifier
                    else torch.flatten(hidden, end_dim=1)
                )
                y = torch.flatten(y)
                hiddens.append(hidden)
                labels.append(y)

            hiddens = torch.cat(hiddens, dim=0)
            labels = torch.cat(labels, dim=0)

            for cls in labels.unique():
                num_batch_classes = (labels == cls).long().sum()

                if num_batch_classes == 0:
                    continue

                self.mu[cls] = hiddens[labels == cls].mean(dim=0)
                self.Sigma[cls] = torch.FloatTensor(
                    np.cov(hiddens[labels == cls].T.numpy())
                ) * (num_batch_classes - 1)

                self.determinants[cls] = torch.det(
                    self.Sigma[cls, :, :]
                )  # Compute determinant
                self.Sigma[cls, :, :] = torch.linalg.inv(self.Sigma[cls, :, :])

    def gmm_predict(self, input_: torch.LongTensor) -> torch.FloatTensor:
        """
        Make a prediction with the Gaussian mixture Model for a batch of inputs.

        Parameters
        ----------
        input_: torch.LongTensor
            Batch of inputs in the form of indexed tokens.

        Returns
        -------
        torch.FloatTensor
            Probability of the input under every mixture component, with one component per class.
        """
        batch_size = input_.shape[0]
        hidden = self.get_hidden(input_)  # batch_size x seq_length x input_size
        hidden = (
            self.get_sequence_representation(hidden).squeeze(1)
            if self.is_sequence_classifier
            else rearrange(hidden, "b s i -> (b s) i")
        )

        hidden = hidden.unsqueeze(1)  # (batch_size x seq_length) x 1 x input_size
        hidden = hidden.repeat(1, self.output_size, 1)
        diff = hidden - self.mu  # (batch_size x seq_length) x output_size x input_size
        diff_t = rearrange(diff, "b o i -> b i o")

        probs = (
            1
            / (2 * math.pi * self.determinants + 1e-6)
            * torch.exp(
                -0.5 * torch.einsum("boi,oii,bio->bo", diff, self.Sigma, diff_t)
            )
        )  # (batch_size x seq_length) x output_size
        probs = rearrange(probs, "(b s) o -> b s o", b=batch_size)

        return probs

    def get_uncertainty(
        self, input_: torch.LongTensor, metric_name: Optional[str] = None
    ) -> torch.FloatTensor:
        """
        Get the uncertainty scores for the current batch.

        Parameters
        ----------
        input_: torch.LongTensor
            (Batch of) Indexed input sequences.
        metric_name: str
            Name of uncertainty metric being used.

        Returns
        -------
        torch.FloatTensor
            Uncertainty scores for the current batch.
        """
        if metric_name is None:
            metric_name = self.default_uncertainty_metric

        if metric_name == "log_prob":
            with torch.no_grad():
                return self.gmm_predict(input_)

        else:
            super().get_uncertainty(input_, metric_name)


class DDUTransformer(Model):
    def __init__(
        self,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        model_dir: Optional[str] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            "ddu_transformer",
            DDUTransformerModule,
            model_params,
            train_params,
            model_dir,
            device,
        )

    def _finetune(
        self,
        data_split: DataSplit,
        verbose: bool,
        summary_writer: Optional[SummaryWriter] = None,
    ):
        """
        As an additional step after training, DDU fits a Gaussian Discriminant Analysis model to
        the training data.

        Parameters
        ----------
        data_split: DataSplit
            Data the GDA is fit on.
        verbose: bool
            Whether to display information about current loss.
        summary_writer: Optional[SummaryWriter]
            Summary writer to track training statistics. Training and validation loss (if applicable) are tracked by
            default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        self.module.eval()  # Disable dropout
        self.module.gmm_fit(data_split)
