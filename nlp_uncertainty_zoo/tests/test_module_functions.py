"""
Make sure that all functions for available models work as expected and are implemented correctly. Specifically,
this concerned the following points:
    - Functions such as get_logits(), predict(), get_sequence_representation()
    - Uncertainty metrics
    - All the above for both LanguageModelingDataset and SequenceClassificationDataset

Most importantly, this *doesn't* include testing the correctness of models. It rather aims to guarantee consistency
across model implementations.
"""

# STD
from abc import ABC, abstractmethod
from typing import Generator
import unittest

# EXT
import torch
from tqdm import tqdm

# PROJECT
from nlp_uncertainty_zoo.config import (
    AVAILABLE_MODELS,
    MODEL_PARAMS,
    TRAIN_PARAMS,
)
from nlp_uncertainty_zoo.datasets import (
    DataSplit,
    LanguageModelingDataset,
    SequenceClassificationDataset,
)
from nlp_uncertainty_zoo.models.model import Model, MultiPredictionMixin
from nlp_uncertainty_zoo.models import TransformerModule

# CONST
# Specify the datasets whose parameters are going to be used to initialize models. The datasets themselves will not be
# used.
TAKE_LANGUAGE_MODELING_HYPERPARAMS_FROM = "ptb"
TAKE_SEQUENCE_CLASSIFICATION_HYPERPARAMS_FROM = "clinc"
# Constants used for testing
NUM_BATCHES = 4
NUM_TYPES = 30
NUM_CLASSES = 6
NUM_PREDICTIONS = 5
BATCH_SIZE = 4
SEQUENCE_LENGTH = 12


class MockDataset:
    """
    Create a mock language modeling dataset.
    """

    def __init__(
        self, num_batches: int, num_types: int, batch_size: int, sequence_length: int
    ):
        self.num_batches = num_batches
        self.num_types = num_types
        self.num_classes = num_types
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.batched_sequences = self.generate_batched_sequences()
        self.fake_split = DataSplit(self.batched_sequences)

        # Use this fake split for all splits
        self._train, self._valid, self._test = (
            self.fake_split,
            self.fake_split,
            self.fake_split,
        )
        self.mock_input = self.batched_sequences[0][0]

    @abstractmethod
    def generate_batched_sequences(self):
        pass


class MockLanguageModelingDataset(LanguageModelingDataset, MockDataset):
    """
    Create a language modeling dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__("", "", {}, BATCH_SIZE)
        MockDataset.__init__(self, *args, **kwargs)

    def generate_batched_sequences(self):
        batches = [
            torch.randint(
                high=self.num_types, size=(self.batch_size, self.sequence_length + 1)
            )
            for _ in range(self.num_batches)
        ]
        return [
            (batches[i][:, :-1], batches[i][:, 1:]) for i in range(self.num_batches)
        ]


class MockSequenceClassificationDataset(SequenceClassificationDataset, MockDataset):
    """
    Create a mock sequence classification dataset.
    """

    def __init__(
        self,
        num_batches: int,
        num_types: int,
        num_classes: int,
        batch_size: int,
        sequence_length: int,
    ):
        self.num_classes = num_classes

        super().__init__("", "", {}, BATCH_SIZE)
        MockDataset.__init__(self, num_batches, num_types, batch_size, sequence_length)

    def generate_batched_sequences(self):
        return [
            (
                torch.randint(
                    high=self.num_types, size=(self.batch_size, self.sequence_length)
                ),
                torch.randint(high=self.num_classes, size=(self.batch_size,)),
            )
            for _ in range(self.num_batches)
        ]


class AbstractFunctionTests(unittest.TestCase, ABC):
    """
    Abstract base class, implementing all tests to check important model functions and their consistency across \
    implemented models.
    """

    mock_dataset = None
    dataset_name = None
    logit_shape = None
    logit_multi_shape = None
    uncertainty_scores_shape = None

    @property
    def trained_models(self) -> Generator[Model, None, None]:
        """
        Returns a generator of trained models to avoid having to hold all trained models in memory.

        Returns
        -------
        Generator[Model]
            Generator that returns one of the available models in trained form during every iteration.
        """

        def _init_and_train_model(model_name: str) -> Model:

            model_params = MODEL_PARAMS[self.dataset_name][model_name]
            train_params = TRAIN_PARAMS[self.dataset_name][model_name]

            # Change some parameters to fit the test environment
            train_params["num_epochs"] = 1
            model_params["vocab_size"] = self.mock_dataset.num_types
            model_params["output_size"] = self.mock_dataset.num_classes
            model_params["is_sequence_classifier"] = isinstance(
                self.mock_dataset, MockSequenceClassificationDataset
            )

            if "sequence_length" in model_params:
                model_params["sequence_length"] = self.mock_dataset.sequence_length

            # Init and fit model
            model = AVAILABLE_MODELS[model_name](model_params, train_params)
            model.fit(dataset=self.mock_dataset, verbose=False)

            return model

        return (
            _init_and_train_model(model_name=model_name)
            for model_name in list(AVAILABLE_MODELS.keys())
        )

    def test_all(self):
        """
        Test all important functionalities of all models for consistency. Check the called functions for more details.
        """
        if self.dataset_name is None:
            return

        with tqdm(total=len(AVAILABLE_MODELS)) as progress_bar:
            for model_class, trained_model in zip(
                AVAILABLE_MODELS.values(), self.trained_models
            ):
                progress_bar.set_description(f'Testing model "{model_class.__name__}"')

                self._test_module_functions(trained_model)
                self._test_uncertainty_metrics(trained_model)

                progress_bar.update(1)

    def _test_module_functions(self, model: Model):
        """
        Test all functions implemented in the Module base class.
        """
        # Pick the right expected logit shape depending on whether model produces multiple predictions
        logit_shape = (
            self.logit_shape
            if not isinstance(model.module, MultiPredictionMixin)
            else self.logit_multi_shape
        )
        mock_input = self.mock_dataset.mock_input

        # Test get_logits()
        logits = model.module.get_logits(mock_input, num_predictions=NUM_PREDICTIONS)
        self.assertTrue(logits.shape == logit_shape)

        # Test predict()
        predictions_module = model.module.predict(mock_input)
        self.assertTrue(predictions_module.shape == self.logit_shape)
        self.assertTrue(
            torch.allclose(
                predictions_module.sum(dim=-1), torch.ones(BATCH_SIZE, SEQUENCE_LENGTH)
            )
        )

        predictions_model = model.predict(mock_input)
        self.assertTrue(predictions_model.shape == self.logit_shape)
        self.assertTrue(
            torch.allclose(
                predictions_model.sum(dim=-1), torch.ones(BATCH_SIZE, SEQUENCE_LENGTH)
            )
        )

        # Test get_sequence_representation()
        hidden, target_size = self._generate_hidden_and_target(
            model, BATCH_SIZE, SEQUENCE_LENGTH
        )
        seq_repr = model.module.get_sequence_representation(hidden)
        self.assertTrue(seq_repr.shape == target_size)

    def _test_uncertainty_metrics(self, model: Model):
        """
        Test all implemented uncertainty metrics, calling them from both the Module and Model class.
        """
        mock_input = self.mock_dataset.mock_input

        # Test default uncertainty metric
        uncertainty_scores_model = model.get_uncertainty(
            mock_input, num_prediction=NUM_PREDICTIONS
        )
        self.assertTrue(uncertainty_scores_model.shape == self.uncertainty_scores_shape)

        uncertainty_scores_module = model.module.get_uncertainty(
            mock_input, num_prediction=NUM_PREDICTIONS
        )
        self.assertTrue(
            uncertainty_scores_module.shape == self.uncertainty_scores_shape
        )

        # Test all available uncertainty metrics through Model and Module
        metrics = list(model.module.single_prediction_uncertainty_metrics) + list(
            model.module.multi_prediction_uncertainty_metrics
        )
        for metric_name in metrics:
            uncertainty_scores_model = model.get_uncertainty(
                mock_input, metric_name, num_prediction=NUM_PREDICTIONS
            )
            self.assertTrue(
                uncertainty_scores_model.shape == self.uncertainty_scores_shape
            )

            uncertainty_scores_module = model.module.get_uncertainty(
                mock_input, metric_name, num_prediction=NUM_PREDICTIONS
            )
            self.assertTrue(
                uncertainty_scores_module.shape == self.uncertainty_scores_shape
            )

    @staticmethod
    def _generate_hidden_and_target(model: Model, batch_size: int, sequence_length):
        repr_size = (
            model.module.hidden_size
            if not isinstance(model.module, TransformerModule)
            else model.module.input_size
        )
        return torch.randn((batch_size, sequence_length, repr_size)), torch.Size(
            (batch_size, 1, repr_size)
        )


class LanguageModelingFunctionTests(AbstractFunctionTests):
    """
    Test all important model functionalities for a language modeling dataset.
    """

    def setUp(self) -> None:
        self.mock_dataset = MockLanguageModelingDataset(
            num_batches=NUM_BATCHES,
            num_types=NUM_TYPES,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        self.dataset_name = TAKE_LANGUAGE_MODELING_HYPERPARAMS_FROM

        # Target shapes
        self.logit_shape = torch.Size((BATCH_SIZE, SEQUENCE_LENGTH, NUM_TYPES))
        self.logit_multi_shape = torch.Size(
            (BATCH_SIZE, NUM_PREDICTIONS, SEQUENCE_LENGTH, NUM_TYPES)
        )
        self.uncertainty_scores_shape = torch.Size((BATCH_SIZE, SEQUENCE_LENGTH))


# TODO: Re-add this
'''
class SequenceClassificationFunctionTests(AbstractFunctionTests, unittest.TestCase):
    """
    Test all important model functionalities for a sequuence classification dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ...  # TODO: Define target shapes

    def setUp(self) -> None:
        self.mock_dataset = MockSequenceClassificationDataset(
            num_batches=NUM_BATCHES,
            num_types=NUM_TYPES,
            num_classes=NUM_CLASSES,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH
        )
        self.dataset_name = TAKE_SEQUENCE_CLASSIFICATION_HYPERPARAMS_FROM
'''
