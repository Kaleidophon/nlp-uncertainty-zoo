"""
Module to implement data reading and batching functionalities.
"""

# STD
from abc import ABC
import codecs
import os
from typing import Dict, Optional, Any

# EXT
from t2i import T2I
import torch
from torch.utils.data import Dataset, TensorDataset


# TODO: How to return just sequences or sequences and labels
# TODO: Language modelling-style batching
# TODO: Create wikitext dataset


class TextDataset(ABC):
    """
    Dataset superclass implementing most central functions like data loading, indexing and batching.
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        splits: Dict[str, str],
        batch_style: str = "padding",
        max_length: Optional[int] = None,
        **indexing_params: Dict[str, Any]
    ):
        """
        Initialize a dataset.

        Parameters
        ----------
        name: str
            Name of dataset. Also used to locate the data for this specific dataset.
        data_dir: str
            Directory in which datasets are located.
        splits: Dict[str, str]
            Dictionary mapping from split ('train', 'valid', 'test') to filename containing that split.
        batch_style: str
            Style used to pack sequences into batches, either 'padding' (pad sequences up to a certain length) or
            'continuous' (style often used for language modelling, sequences continue over batch boundaries). Default is
            'padding'.
        max_length: Optional[int]
            Maximum length for padding if batch_style='padding'. No padding if max_length=None, which is the default.
        indexing_params: Dict[str, Any]
            Parameters for t2i.T2I indexing class.
        """

        assert batch_style in ("padding", "continuous")

        self.name = name
        self.data_dir = data_dir
        self.splits = splits
        self.t2i = None
        self.indexing_params = indexing_params
        self.batch_style = batch_style
        self.max_length = max_length

        # Splits
        self._train = None
        self._valid = None
        self._test = None

    @property
    def train(self) -> Dataset:
        """
        Retrieve the training split of a dataset.

        Returns
        -------
        Dataset
            Training split.
        """
        if self._train is None:
            self._train = self._load("train")

        return self._train

    @property
    def valid(self) -> Dataset:
        """
        Retrieve the validation split of a dataset.

        Returns
        -------
        Dataset
            Validation split.
        """
        # If train split hasn't been created yet, do that first to maintain consistent indexing
        if self._train is None:
            self.train

        if self._valid is None:
            self._valid = self._load("valid")

        return self._valid

    @property
    def test(self) -> Dataset:
        """
        Retrieve the test split of a dataset.

        Returns
        -------
        Dataset
            Test split.
        """
        # If train split hasn't been created yet, do that first to maintain consistent indexing
        if self._train is None:
            self.train

        if self._test is None:
            self._test = self._load("test")

        return self._test

    def _load(self, split: str) -> Dataset:
        """
        Load a data split as well as index and batch it. If the split is not "train" and the training split hasn't been
        loaded yet, it will be to build the index.

        Parameters
        ----------
        split: str
            Name of the split. Has to be either 'train', 'valid' or 'test'.

        Returns
        -------
        Dataset
            Return the loaded split.
        """
        assert split in ("train", "test", "valid")

        split_paths = os.path.join(self.data_dir, self.name, self.splits[split])

        with codecs.open(split_paths, "rb", "utf-8") as file:
            lines = [line.strip() for line in file.readlines()]

        # Initialize indexing
        if split == "train" and self.t2i is None:
            self.t2i = T2I.build(lines, **self.indexing_params)

        # Index sentences and convert to tensors
        indexed_lines = list(
            map(
                torch.LongTensor,
                self.t2i(
                    lines,
                    pad_to=self.max_length if self.batch_style == "padding" else None,
                ),
            )
        )

        return TensorDataset(*indexed_lines)
