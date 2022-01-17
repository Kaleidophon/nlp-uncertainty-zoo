"""
Module to implement data reading and batching functionalities.
"""

# STD
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

# EXT
from sklearn.preprocessing import LabelEncoder
import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


class DatasetBuilder(ABC):
    """
    Abstract dataset builder class used to create a variety of different dataset types, including sequence prediction,
    token prediction, next-token-prediction language modelling and masked language modelling.
    """

    def __init__(
        self,
        name: str,
        splits: Dict[str, Any],
        type_: str,
        tokenizer: PreTrainedTokenizerBase,
    ):
        """
        Initialize a DatasetBuilder.

        Parameters
        ----------
        name: str
            Name of the final dataset.
        splits: Dict[str, Any]
            Dictionary pointing to the files containing the training, validation and test split.
        type_: str
            String that further specifies the type of dataset being built (see LanguageModellingDatasetBuilder and
            ClassificationDatasetBuilder for more detail).
        tokenizer: PreTrainedTokenizerBase
            Pre-trained tokenizer.
        """
        self.name = name
        self.splits = splits
        self.type = type_
        self.tokenizer = tokenizer
        self.dataset = None

    @abstractmethod
    def build(self) -> Dataset:
        """
        Build a dataset.

        Returns
        -------
        Dataset:
            Ready-to-use dataset.
        """
        pass


class LanguageModellingDatasetBuilder(DatasetBuilder):
    ...  # TODO


class ClassificationDatasetBuilder(DatasetBuilder):
    ...  # TODO


class PennTreebankBuilder(LanguageModellingDatasetBuilder):
    """
    Dataset class for the Penn Treebank.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        sequence_length: int,
        **indexing_params: Dict[str, Any],
    ):
        super().__init__(
            name="ptb",
            data_dir=data_dir,
            splits={
                "train": "ptb.train.txt",
                "valid": "ptb.valid.txt",
                "test": "ptb.test.txt",
            },
            batch_size=batch_size,
            batch_style="continuous",
            sequence_length=sequence_length,
            **indexing_params,
        )


class ClincBuilder(ClassificationDatasetBuilder):
    """
    Dataset class for the CLINC OOS dataset.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        sequence_length: int,
        **indexing_params: Dict[str, Any],
    ):
        super().__init__(
            name="clinc",
            data_dir=data_dir,
            splits={
                "train": "train.csv",
                "valid": "val.csv",
                "test": "test.csv",
                "oos_test": "oos_test.csv",
            },
            batch_size=batch_size,
            sequence_length=sequence_length,
            **indexing_params,
        )
