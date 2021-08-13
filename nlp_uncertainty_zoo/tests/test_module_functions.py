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
from nlp_uncertainty_zoo.models.model import Model

# CONST
# Specify the datasets whose parameters are going to be used to initialize models. The datasets themselves will not be
# used.
TAKE_LANGUAGE_MODELING_HYPERPARAMS_FROM = "ptb"
TAKE_SEQUENCE_CLASSIFICATION_HYPERPARAMS_FROM = "clinc"
# Constants used for testing
NUM_BATCHES = 4
NUM_TYPES = 30
NUM_CLASSES = 6
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


class AbstractFunctionTests(ABC):
    """
    Abstract base class, implementing all tests to check important model functions and their consistency across \
    implemented models.
    """

    mock_dataset = None
    dataset_name = None

    @property
    def trained_models(self) -> Generator[Model, None, None]:
        """
        Returns a generator of trained models to avoid having to hold all trained models in memory.

        Returns
        -------
        Generator[Model]
            Generator that returns one of the available models in trained form during every iteration.
        """

        def _init_and_train_model(model_name) -> Model:
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
            for model_name in AVAILABLE_MODELS.keys()
        )

    def test_all(self):
        """
        Test all important functionalities of all models for consistency. Check the called functions for more details.
        """
        for model in self.trained_models:
            self._test_module_functions(model)
            self._test_model_functions(model)
            self._test_uncertainty_metrics(model)

    def _test_module_functions(self, model: Model):
        """
        Test all functions implemented in the Module base class.
        """
        ...  # TODO

    def _test_model_functions(self, model: Model):
        """
        Test all functions implemented in the Model base class.
        """
        ...  # TODO

    def _test_uncertainty_metrics(self, model: Model):
        """
        Test all implemented uncertainty metrics, calling them from both the Module and Model class.
        """
        ...  # TODO


class LanguageModelingFunctionTests(AbstractFunctionTests, unittest.TestCase):
    """
    Test all important model functionalities for a language modeling dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ...  # TODO: Define target shapes

    def setUp(self) -> None:
        self.mock_dataset = MockLanguageModelingDataset(
            num_batches=NUM_BATCHES,
            num_types=NUM_TYPES,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        self.dataset_name = TAKE_LANGUAGE_MODELING_HYPERPARAMS_FROM


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
