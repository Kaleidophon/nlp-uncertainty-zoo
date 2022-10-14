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
from abc import ABC
from typing import Generator, Dict, Tuple
import unittest

# EXT
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# PROJECT
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS, DEFAULT_PARAMS
from nlp_uncertainty_zoo.models.model import Model, MultiPredictionMixin
from nlp_uncertainty_zoo.models import TransformerModule

# CONST
TAKE_LANGUAGE_MODELING_HYPERPARAMS_FROM = "language_modelling"
TAKE_SEQUENCE_CLASSIFICATION_HYPERPARAMS_FROM = "sequence_classification"
# Constants used for testing
NUM_INSTANCES = 16
NUM_TYPES = 30
NUM_CLASSES = 6
NUM_PREDICTIONS = 5
BATCH_SIZE = 4
SEQUENCE_LENGTH = 12
NUM_TRAINING_STEPS = 2


class MockDatasetBuilder(ABC):
    """
    Create a mock language modeling dataset.
    """

    def __init__(self, num_instances: int, num_types: int, sequence_length: int):
        self.num_instances = num_instances
        self.num_types = num_types
        self.sequence_length = sequence_length
        self.mock_input = None


class MockLanguageModellingDatasetBuilder(MockDatasetBuilder):
    def __init__(self, num_instances: int, num_types: int, sequence_length: int):
        super().__init__(num_instances, num_types, sequence_length)
        self.num_classes = num_types

    def build(self, batch_size: int) -> Dict[str, DataLoader]:
        sequences = torch.multinomial(
            torch.ones(self.num_types),
            self.num_instances * self.sequence_length,
            replacement=True,
        )
        sequences = sequences.reshape(self.num_instances, self.sequence_length)
        labels = sequences.clone()
        masked = torch.randint(1, (self.num_instances, self.sequence_length)).bool()
        labels[masked] = -100
        instances = [
            {
                "labels": labels[i, :],
                "input_ids": sequences[i, :],
                "attention_mask": torch.ones((self.sequence_length,)),
            }
            for i in range(self.num_instances)
        ]
        dataloader = DataLoader(instances, batch_size=batch_size)
        self.mock_input = next(iter(dataloader))

        return {"train": dataloader}


class MockSequenceClassificationDatasetBuilder(MockDatasetBuilder):
    def __init__(
        self, num_instances: int, num_types: int, sequence_length: int, num_classes: int
    ):
        super().__init__(num_instances, num_types, sequence_length)
        self.num_classes = num_classes

    def build(self, batch_size: int) -> Dict[str, DataLoader]:
        sequences = torch.multinomial(
            torch.ones(self.num_types),
            self.num_instances * self.sequence_length,
            replacement=True,
        )
        sequences = sequences.reshape(self.num_instances, self.sequence_length)
        labels = torch.randint(self.num_classes, (self.num_instances,))
        instances = [
            {
                "labels": labels[i],
                "input_ids": sequences[i, :],
                "attention_mask": torch.ones((self.sequence_length,)),
            }
            for i in range(self.num_instances)
        ]
        dataloader = DataLoader(instances, batch_size=batch_size)
        self.mock_input = next(iter(dataloader))

        return {"train": dataloader}


class AbstractFunctionTests(unittest.TestCase, ABC):
    """
    Abstract base class, implementing all tests to check important model functions and their consistency across \
    implemented models.
    """

    mock_dataset_builder = None
    mock_dataset = None
    dataset_name = None
    logit_shape = None
    logit_multi_shape = None
    uncertainty_scores_shape = None

    @property
    def trained_models(self) -> Generator[Tuple[str, Model], None, None]:
        """
        Returns a generator of trained models to avoid having to hold all trained models in memory.

        Returns
        -------
        Generator[Model]
            Generator that returns one of the available models in trained form during every iteration.
        """

        def _init_and_train_model(model_name: str) -> Tuple[str, Model]:

            model_params = DEFAULT_PARAMS[self.dataset_name][model_name]
            mock_dataset = self.mock_dataset_builder.build(BATCH_SIZE)

            # Change some parameters to fit the test environment
            model_params["num_training_steps"] = NUM_TRAINING_STEPS
            model_params["vocab_size"] = self.mock_dataset_builder.num_types
            model_params["output_size"] = self.mock_dataset_builder.num_classes
            model_params["is_sequence_classifier"] = isinstance(
                self.mock_dataset_builder, MockSequenceClassificationDatasetBuilder
            )
            model_params["num_predictions"] = NUM_PREDICTIONS

            if "sequence_length" in model_params:
                model_params[
                    "sequence_length"
                ] = self.mock_dataset_builder.sequence_length

            # Init and fit model
            model = AVAILABLE_MODELS[model_name](model_params)
            model.fit(train_split=mock_dataset["train"], verbose=False)

            return model_name, model

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
            for model_name, trained_model in self.trained_models:
                progress_bar.set_description(f'Testing model "{model_name}"')
                progress_bar.update(1)

                self._test_module_functions(trained_model)
                self._test_uncertainty_metrics(trained_model)

                del trained_model  # Free up memory

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
        mock_input = self.mock_dataset_builder.mock_input

        # Test get_logits()
        logits = model.module.get_logits(
            mock_input["input_ids"],
            attention_mask=mock_input["attention_mask"],
            num_predictions=NUM_PREDICTIONS,
        )
        self.assertTrue(logits.shape == logit_shape)

        # Test predict()
        predictions_module = model.module.predict(
            mock_input["input_ids"], attention_mask=mock_input["attention_mask"]
        )
        self.assertTrue(predictions_module.shape == self.logit_shape)
        self.assertTrue(
            torch.allclose(
                predictions_module.sum(dim=-1), torch.ones(BATCH_SIZE, SEQUENCE_LENGTH)
            )
        )

        predictions_model = model.predict(
            mock_input["input_ids"], attention_mask=mock_input["attention_mask"]
        )
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
        mock_input = self.mock_dataset_builder.mock_input

        # Test default uncertainty metric
        uncertainty_scores_model = model.get_uncertainty(
            mock_input["input_ids"],
            attention_mask=mock_input["attention_mask"],
            num_prediction=NUM_PREDICTIONS,
        )
        self.assertTrue(uncertainty_scores_model.shape == self.uncertainty_scores_shape)

        uncertainty_scores_module = model.module.get_uncertainty(
            mock_input["input_ids"],
            attention_mask=mock_input["attention_mask"],
            num_prediction=NUM_PREDICTIONS,
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
                mock_input["input_ids"],
                metric_name=metric_name,
                attention_mask=mock_input["attention_mask"],
                num_prediction=NUM_PREDICTIONS,
            )
            self.assertTrue(
                uncertainty_scores_model.shape == self.uncertainty_scores_shape
            )

            uncertainty_scores_module = model.module.get_uncertainty(
                mock_input["input_ids"],
                metric_name=metric_name,
                attention_mask=mock_input["attention_mask"],
                num_prediction=NUM_PREDICTIONS,
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
        self.mock_dataset_builder = MockLanguageModellingDatasetBuilder(
            num_types=NUM_TYPES,
            sequence_length=SEQUENCE_LENGTH,
            num_instances=NUM_INSTANCES,
        )
        self.dataset_name = TAKE_LANGUAGE_MODELING_HYPERPARAMS_FROM

        # Target shapes
        self.logit_shape = torch.Size((BATCH_SIZE, SEQUENCE_LENGTH, NUM_TYPES))
        self.logit_multi_shape = torch.Size(
            (BATCH_SIZE, NUM_PREDICTIONS, SEQUENCE_LENGTH, NUM_TYPES)
        )
        self.uncertainty_scores_shape = torch.Size((BATCH_SIZE, SEQUENCE_LENGTH))


class SequenceClassificationFunctionTests(AbstractFunctionTests):
    """
    Test all important model functionalities for a sequuence classification dataset.
    """

    def setUp(self) -> None:
        self.mock_dataset_builder = MockSequenceClassificationDatasetBuilder(
            num_types=NUM_TYPES,
            num_classes=NUM_CLASSES,
            sequence_length=SEQUENCE_LENGTH,
            num_instances=NUM_INSTANCES,
        )
        self.dataset_name = TAKE_SEQUENCE_CLASSIFICATION_HYPERPARAMS_FROM

        # Target shapes
        self.logit_shape = torch.Size((BATCH_SIZE, 1, NUM_CLASSES))
        self.logit_multi_shape = torch.Size(
            (BATCH_SIZE, NUM_PREDICTIONS, 1, NUM_CLASSES)
        )
        self.uncertainty_scores_shape = torch.Size((BATCH_SIZE, 1))
