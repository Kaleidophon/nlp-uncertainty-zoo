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
import unittest

# EXT
import torch

# PROJECT
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS, AVAILABLE_DATASETS
from nlp_uncertainty_zoo.datasets import (
    LanguageModelingDataset,
    SequenceClassificationDataset,
    DataSplit,
)
from nlp_uncertainty_zoo.utils.types import BatchedSequences

# CONST
# Specify the datasets whose parameters are going to be used to initialize models. The datasets themselves will not be
# used.
TAKE_LANGUAGE_MODELING_HYPERPARAMS_FROM = "ptb"
TAKE_SEQUENCE_CLASSIFICATION_HYPERPARAMS_FROM = "clinc"


class MockDataset:
    """
    Create a mock language modeling dataset.
    """

    def __init__(
        self, num_batches: int, num_types: int, batch_size: int, sequence_length: int
    ):
        self.num_batches = num_batches
        self.num_types = num_types
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.batched_sequences = self.generate_batched_sequences()
        self.fake_split = DataSplit(self.batched_sequences)

        # Use this fake split for all splits
        for split in ["train", "valid", "test"]:
            setattr(self, split, self.fake_split)

    @abstractmethod
    def generate_batched_sequences(self):
        pass


class MockLanguageModelingDataset(MockDataset):
    """
    Create a language modeling dataset.
    """

    def generate_batched_sequences(self):
        batches = [
            torch.randint(
                high=self.num_types, shape=(self.batch_size, self.sequence_length + 1)
            )
            for _ in range(self.num_batches)
        ]
        return [
            (batches[i][:, :-1], batches[i][:, 1:]) for i in range(self.num_batches)
        ]


class MockSequenceClassificationDataset(MockDataset):
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
        super().__init__(num_batches, num_types, batch_size, sequence_length)

    def generate_batched_sequences(self):
        return [
            (
                torch.randint(
                    high=self.num_types, shape=(self.batch_size, self.sequence_length)
                ),
                torch.randint(high=self.num_classes, shape=(self.batch_size,)),
            )
            for _ in range(self.num_batches)
        ]


class AbstractFunctionTests(ABC):
    """
    Abstract base class, implementing all tests to check important model functions and their consistency across \
    implemented models.
    """

    def init_and_train_models(self) -> None:
        ...  # TODO: Init and train models

    def test_module_functions(self):
        """
        Test all functions implemented in the Module base class.
        """
        ...  # TODO

    def test_model_functions(self):
        """
        Test all functions implemented in the Model base class.
        """
        ...  # TODO

    def test_uncertainty_metrics(self):
        """
        Test all implemented uncertainty metrics.
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
        ...  # TODO: Init dataset
        ...  # TODO: Init models


class SequenceClassificationFunctionTests(AbstractFunctionTests, unittest.TestCase):
    """
    Test all important model functionalities for a sequuence classification dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ...  # TODO: Define target shapes

    def setUp(self) -> None:
        ...  # TODO: Init dataset
        ...  # TODO: Init models
