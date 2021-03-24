"""
Module to implement data reading and batching functionalities.
"""

# STD
from abc import ABC, abstractmethod
import codecs
from copy import deepcopy
import math
import os
from random import shuffle
from typing import Dict, Optional, Any, List, Tuple

# EXT
from t2i import T2I
import torch
from torch.utils.data import Dataset

# PROJECT
from src.types import BatchedSequences, Device


class DataSplit(Dataset):
    """
    Wrapper class for a data split.
    """

    def __init__(self, batched_sequences: BatchedSequences):
        self.batched_sequences = batched_sequences

    def __len__(self):
        return len(self.batched_sequences)

    def __getitem__(self, item):
        return self.batched_sequences[item]

    def __iter__(self):
        for input_, labels in self.batched_sequences:
            yield input_, labels

    def to(self, device: Device):
        self.batched_sequences = [
            (input_.to(device), target.to(device))
            for input_, target in self.batched_sequences
        ]

    def shuffle(self):
        dataset = deepcopy(self)
        shuffle(dataset.batched_sequences)

        return dataset


class TextDataset(ABC):
    """
    Dataset superclass implementing most central functions like data loading, indexing and batching.
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        splits: Dict[str, str],
        batch_size: int,
        batch_style: str = "padding",
        sequence_length: Optional[int] = None,
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
        batch_size: int
            Number of sequences in a batch.
        batch_style: str
            Style used to pack sequences into batches, either 'padding' (pad sequences up to a certain length) or
            'continuous' (style often used for language modelling, sequences continue over batch boundaries). Default is
            'padding'.
        sequence_length: Optional[int]
            Maximum length for padding if batch_style='padding'. No padding if max_length=None, which is the default.
        indexing_params: Dict[str, Any]
            Parameters for t2i.T2I indexing class.
        """

        assert batch_style in ("padding", "continuous")

        self.name = name
        self.data_dir = data_dir
        self.splits = splits
        self.batch_size = batch_size
        self.t2i = None
        self.indexing_params = indexing_params
        self.batch_style = batch_style
        self.sequence_length = sequence_length

        # Splits
        self._train = None
        self._valid = None
        self._test = None

    @property
    def train(self) -> DataSplit:
        """
        Retrieve the training split of a dataset.

        Returns
        -------
        Dataset
            Training split.
        """
        if self._train is None:
            self._train = self._load("train")

        batches = self._train

        if self.batch_style == "padding":
            batches.shuffle()

        return batches

    @property
    def valid(self) -> DataSplit:
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
    def test(self) -> DataSplit:
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

    def _load(self, split: str) -> DataSplit:
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
            sequences = [line.strip() for line in file.readlines()]

        # Initialize indexing
        if split == "train" and self.t2i is None:
            self.t2i = T2I.build(sequences, **self.indexing_params)

        return self._batch(split, sequences)

    @abstractmethod
    def _get_labels(
        self, split: str, batched_sequences: BatchedSequences
    ) -> Tuple[BatchedSequences, List[torch.LongTensor]]:
        """
        Get the labels for a dataset by loading them or deriving them from the inputs.

        Parameters
        ----------
        split: str
            Name of the split. Has to be either 'train', 'valid' or 'test'.
        batched_sequences: BatchedSequences
            Batched sequences in a data split.

        Returns
        -------
        Tuple[List[torch.LongTensor], List[torch.LongTensor]]
            Paired list of batched inputs and labels.
        """
        pass

    def _batch(self, split: str, sequences: List[str]) -> DataSplit:
        """
        Index and batch sequences based on batching style specified by batch_style. If batch_style="padding", one batch
        instance will correspond to a single sequences padded up to a certain length either specified by sequence_length
        or corresponding to the maximum sequence length found in the training set.

        Parameters
        ----------
        split: str
            Data split to be batched.
        sequences: List[str]
            List of sequences in split as strings.

        Returns
        -------
        DataSplit
            DataSplit class containing indexed sequences and target labels.
        """
        batched_sequences = None

        if self.batch_style == "padding":

            indexed_sequences = list(
                map(
                    torch.LongTensor,
                    self.t2i(
                        sequences,
                        pad_to=self.sequence_length
                        if self.sequence_length is not None
                        else "max",
                    ),
                )
            )

            if self.sequence_length is None:
                self.sequence_length = max(len(seq) for seq in indexed_sequences)

            # Filter out sequences which are too long
            indexed_sequences = filter(
                lambda seq: len(seq) <= self.sequence_length, indexed_sequences
            )

            batched_sequences = torch.stack(
                list(map(torch.LongTensor, indexed_sequences)), dim=0
            )
            batched_sequences = torch.split(
                batched_sequences, split_size_or_sections=self.batch_size, dim=0
            )

        elif self.batch_style == "continuous":
            indexed_sequences = list(map(torch.LongTensor, self.t2i(sequences)))
            indexed_sequences = torch.cat(indexed_sequences, dim=0)

            # Work out how cleanly we can divide the dataset into batch-sized parts
            num_batched_steps = indexed_sequences.shape[0] // self.batch_size

            # Trim off any extra elements that wouldn't cleanly fit (remainders)
            indexed_sequences = indexed_sequences.narrow(
                0, 0, num_batched_steps * self.batch_size
            )

            # Evenly divide the data across the bsz batches.
            raw_batches = indexed_sequences.view(self.batch_size, -1).t().contiguous()

            num_batches = math.ceil(num_batched_steps / self.sequence_length)
            batched_sequences = [
                raw_batches[
                    n * self.sequence_length : (n + 1) * self.sequence_length + 1, :
                ]
                for n in range(num_batches)
            ]

        batched_sequences, batched_labels = self._get_labels(split, batched_sequences)

        # Batches of (sequences, labels)
        return DataSplit(list(zip(batched_sequences, batched_labels)))


class LanguageModelingDataset(TextDataset):
    """
    A superclass for language modeling datasets.
    """

    def _get_labels(
        self, split: str, batched_sequences: BatchedSequences
    ) -> Tuple[List[torch.LongTensor], List[torch.LongTensor]]:
        """
        Get the labels for a dataset by loading them or deriving them from the inputs.

        Parameters
        ----------
        split: str
            Name of the split. Has to be either 'train', 'valid' or 'test'.
        batched_sequences: BatchedSequences
            Batched sequences in a data split.

        Returns
        -------
        Tuple[List[torch.LongTensor], List[torch.LongTensor]]
            Paired list of batched inputs and labels.
        """
        batched_sequences, batched_labels = zip(
            *[(batch[:, :-1], batch[:, 1:]) for batch in batched_sequences]
        )

        return list(batched_sequences), list(batched_labels)


class Wikitext103Dataset(LanguageModelingDataset):
    """
    Dataset class for the Wikitext-103 dataset.
    """

    def __init__(self, data_dir: str, batch_size: int, sequence_length: int):
        super().__init__(
            name="wikitext-103",
            data_dir=data_dir,
            splits={
                "train": "wiki.train.tokens",
                "valid": "wiki.valid.tokens",
                "test": "wiki.test.tokens",
            },
            batch_size=batch_size,
            batch_style="padding",
            sequence_length=sequence_length,
        )
