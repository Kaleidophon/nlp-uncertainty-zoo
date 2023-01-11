"""
Implementation of the Deep Deterministic Uncertainty (DDU) Transformer by
 `Mukhoti et al. (2021) <https://arxiv.org/pdf/2102.11582.pdf>`_.
"""

# STD
import math
from typing import Optional, Dict, Any, List, Type

# EXT
import numpy as np
from sklearn.decomposition import PCA
import torch
from einops import rearrange
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import transformers

# PROJECT
from nlp_uncertainty_zoo.models.spectral import (
    SpectralTransformerModule,
    SpectralBertModule,
)
from nlp_uncertainty_zoo.models.model import Model
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun


class DDUMixin:
    """
    Implementation of the functions used by the Deep Deterministic Uncertainty (DDU) Transformer by
    `Mukhoti et al. (2021) <https://arxiv.org/pdf/2102.11582.pdf>`_. as a Mixin class. This is done to avoid code
    redundancies.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        ignore_indices: List[int],
        projection_size: Optional[int] = None
    ):
        """
        Initialize a DDUMixin.

        Parameters
        ----------
        input_size: int
            Dimensionality of input to the Gaussian Mixture Model layer.
        output_size: int
            Number of classes.
        ignore_indices: List[int]
            List of indices for which activations should be ignored when fitting the GMM.
        projection_size: Optional[int]
            If given, project hidden activations into a subspace with dimensionality projection_size. Default is None.
        """
        self.ignore_indices = ignore_indices
        self.projection_size = projection_size
        self.gda_size = input_size if projection_size is None else projection_size
        self.pca = None

        # Parameters for Gaussian Discriminant Analysis
        self.mu = torch.zeros(
            output_size, self.gda_size
        )
        self.Sigma = torch.stack(
            [torch.eye(self.gda_size, self.gda_size) for _ in range(self.output_size)]
        )
        self.determinants = torch.zeros(output_size)

    def gmm_fit(self, data_split: DataLoader) -> None:
        """
        Fit a Gaussian mixture model on the feature representations of the trained model.

        Parameters
        ----------
        data_split: DataSplit
            Data split used for fitting, usually the training or validation split.
        """

        with torch.no_grad():
            hiddens, all_labels = [], []

            for i, batch in enumerate(data_split):

                attention_mask, input_ids, labels = (
                    batch["attention_mask"].to(self.device),
                    batch["input_ids"].to(self.device),
                    batch["labels"].to(self.device),
                )

                hidden = self.get_hidden(input_ids, attention_mask=attention_mask)

                if not self.is_sequence_classifier:
                    # Filter our labels and activations for uninformative classes like PAD
                    batch_mask = torch.all(
                        torch.stack([input_ids != idx for idx in self.ignore_indices]), dim=0
                    )
                    labels = labels[batch_mask]
                    hidden = hidden[batch_mask]

                else:
                    hidden = (
                        self.get_sequence_representation(hidden).squeeze(1)
                    )

                hiddens.append(hidden)
                all_labels.append(labels)

            hiddens = torch.cat(hiddens, dim=0)
            all_labels = torch.cat(all_labels, dim=0).to(self.device)

            # Do PCA first before fitting to reduce memory usage
            if self.projection_size is not None:
                device = hidden.device
                self.pca = PCA(n_components=self.projection_size)
                self.pca.fit(hiddens.cpu().detach().numpy())
                hiddens = torch.FloatTensor(self.pca.transform(hiddens.cpu().detach().numpy())).to(device)

            for cls in labels.unique():

                if cls == -100:
                    continue

                num_batch_classes = (all_labels == cls).long().sum()

                if num_batch_classes == 0:
                    continue

                self.mu[cls] = hiddens[all_labels == cls].mean(dim=0)
                self.Sigma[cls] = torch.FloatTensor(
                    np.cov(hiddens[all_labels == cls].T.cpu().detach().numpy())
                ).to(self.device) * (num_batch_classes - 1)

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

        if self.pca is not None:
            device = hidden.device
            hidden = torch.FloatTensor(self.pca.transform(hidden.cpu().detach().numpy())).to(device)

        hidden = hidden.unsqueeze(1)  # (batch_size x seq_length) x 1 x input_size
        hidden = hidden.repeat(1, self.output_size, 1)
        diff = hidden - self.mu  # (batch_size x seq_length) x output_size x input_size
        diff_t = rearrange(diff, "b o i -> b i o")

        probs = torch.log(
            1 - (
                1 / (2 * math.pi * self.determinants + 1e-6)
                * torch.exp(
                    -0.5 * torch.einsum("boi,oii,bio->bo", diff, self.Sigma, diff_t)
                )
            ) + 1e-6
        )  # (batch_size x seq_length) x output_size
        probs = rearrange(probs, "(b s) o -> b s o", b=batch_size)

        return probs


class DDUTransformerModule(SpectralTransformerModule, DDUMixin):
    """
    Implementation of the Deep Deterministic Uncertainty (DDU) Transformer by
    `Mukhoti et al. (2021) <https://arxiv.org/pdf/2102.11582.pdf>`_.
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
        spectral_norm_upper_bound: float,
        projection_size: int,
        is_sequence_classifier: bool,
        ignore_indices: List[int],
        device: Device,
        **build_params,
    ):
        """
        Initialize a DDU transformer module.

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
            Input dropout added to embeddings.
        dropout: float
            Dropout rate.
        num_heads: int
            Number of self-attention heads per layer.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings.
        spectral_norm_upper_bound: float
            Set a limit when weight matrices will be spectrally normalized if their eigenvalue surpasses it.
        projection_size: int
            Size hidden dimensions are projected to using PCA to save memory if given.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        ignore_indices: List[int]
            Token indices to ignore when fitting the Gaussian Discriminant Analysis.
        device: Device
            Device the model is located on.
        """
        super().__init__(
            vocab_size=vocab_size,
            output_size=output_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_dropout=input_dropout,
            dropout=dropout,
            num_heads=num_heads,
            sequence_length=sequence_length,
            spectral_norm_upper_bound=spectral_norm_upper_bound,
            is_sequence_classifier=is_sequence_classifier,
            device=device,
        )
        DDUMixin.__init__(self, input_size, output_size, ignore_indices, projection_size)


class DDUTransformer(Model):
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        input_size: int = 512,
        hidden_size: int = 512,
        projection_size: int = 64,
        num_layers: int = 6,
        input_dropout: float = 0.2,
        dropout: float = 0.1,
        num_heads: int = 16,
        sequence_length: int = 128,
        spectral_norm_upper_bound: float = 0.95,
        ignore_indices: List[int] = tuple(),
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
        Initialize a DDU transformer.

        Parameters
        ----------
        vocab_size: int
            Vocabulary size.
        output_size: int
            Size of output of model.
        input_size: int
            Dimensionality of input to model. Default is 512.
        hidden_size: int
            Size of hidden representations. Default is 512.
        projection_size: int
            Size hidden dimensions are projected to using PCA to save memory if given. Default is 64.
        num_layers: int
            Number of model layers. Default is 6.
        input_dropout: float
            Dropout rate. Default is 0.2.
        dropout: float
            Dropout rate. Default is 0.1.
        num_heads: int
            Number of self-attention heads per layer. Default is 16.
        sequence_length: int
            Maximum sequence length in dataset. Used to initialize positional embeddings. Default is 128.
        spectral_norm_upper_bound: float
            Set a limit when weight matrices will be spectrally normalized if their eigenvalue surpasses it. Default is
            0.95.
        ignore_indices: List[int]
            Token indices to ignore when fitting the Gaussian Discriminant Analysis. Is empty by default.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step. Default is True.
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
            "ddu_transformer",
            DDUTransformerModule,
            input_size=input_size,
            output_size=output_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            input_dropout=input_dropout,
            dropout=dropout,
            num_heads=num_heads,
            sequence_length=sequence_length,
            spectral_norm_upper_bound=spectral_norm_upper_bound,
            projection_size=projection_size,
            ignore_indices=ignore_indices,
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

    def _finetune(
        self,
        data_split: DataLoader,
        verbose: bool,
        wandb_run: Optional[WandBRun] = None,
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
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        self.module.eval()  # Disable dropout
        self.module.gmm_fit(data_split)
        self.module.single_prediction_uncertainty_metrics["log_prob"] = self.module.gmm_predict
        self.module.train()


class DDUBertModule(SpectralBertModule, DDUMixin):
    """
    Implementation of the Deep Deterministic Uncertainty (DDU) Transformer by
    `Mukhoti et al. (2021) <https://arxiv.org/pdf/2102.11582.pdf>`_ in the form of a pre-trained BERT.
    """

    def __init__(
        self,
        bert_name: str,
        output_size: int,
        projection_size: int,
        spectral_norm_upper_bound: float,
        is_sequence_classifier: bool,
        ignore_indices: List[int],
        device: Device,
        **build_params,
    ):
        """
        Initialize a DDU Bert Module.

        Parameters
        ----------
        bert_name: str
            Name of the underlying BERT, as specified in HuggingFace transformers.
        output_size: int
            Number of classes.
        projection_size: int
            Size hidden dimensions are projected to using PCA to save memory if given.
        spectral_norm_upper_bound: float
            Set a limit when weight matrices will be spectrally normalized if their eigenvalue surpasses it.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step.
        ignore_indices: List[int]
            Token indices to ignore when fitting the Gaussian Discriminant Analysis. Is empty by default.
        device: Device
            Device the model should be moved to.
        """
        super().__init__(
            bert_name,
            output_size,
            spectral_norm_upper_bound,
            is_sequence_classifier,
            device,
            **build_params,
        )
        DDUMixin.__init__(self, self.bert.config.hidden_size, output_size, ignore_indices, projection_size)

    def get_uncertainty(
        self,
        input_: torch.LongTensor,
        *args,
        metric_name: Optional[str] = None,
        **kwargs,
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
            return super().get_uncertainty(input_, metric_name=metric_name, **kwargs)


class DDUBert(Model):
    def __init__(
        self,
        bert_name: str,
        output_size: int,
        projection_size: int = 64,
        spectral_norm_upper_bound: float = 0.95,
        ignore_indices: List[int] = tuple(),
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
        Initialize a DDU Bert.

        Parameters
        ----------
        bert_name: str
            Name of the underlying BERT, as specified in HuggingFace transformers.
        output_size: int
            Number of classes.
        projection_size: int
            Size hidden dimensions are projected to using PCA to save memory if given.
        spectral_norm_upper_bound: float
            Set a limit when weight matrices will be spectrally normalized if their eigenvalue surpasses it. Default is
            0.95.
        ignore_indices: List[int]
            Token indices to ignore when fitting the Gaussian Discriminant Analysis. Is empty by default.
        is_sequence_classifier: bool
            Indicate whether model is going to be used as a sequence classifier. Otherwise, predictions are going to
            made at every time step. Default is True.
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
            f"ddu-{bert_name}",
            module_class=DDUBertModule,
            bert_name=bert_name,
            output_size=output_size,
            spectral_norm_upper_bound=spectral_norm_upper_bound,
            projection_size=projection_size,
            ignore_indices=ignore_indices,
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

    def _finetune(
        self,
        data_split: DataLoader,
        verbose: bool,
        wandb_run: Optional[WandBRun] = None,
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
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.
        """
        self.module.eval()  # Disable dropout
        self.module.gmm_fit(data_split)
        self.module.single_prediction_uncertainty_metrics["log_prob"] = self.module.gmm_predict
        self.module.train()
