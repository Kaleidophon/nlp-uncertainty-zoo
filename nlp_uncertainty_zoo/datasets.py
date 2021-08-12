"""
Module to implement data reading and batching functionalities.
"""

# STD
from abc import ABC
import codecs
from copy import deepcopy
import math
import os
from random import shuffle
from typing import Dict, Optional, Any, List, Tuple

# EXT
from sklearn.preprocessing import LabelEncoder
from t2i import T2I
import torch
from torch.utils.data import Dataset

# PROJECT
from nlp_uncertainty_zoo.utils.types import BatchedSequences, Device

# TODO: Add WILDS text dataset
# TODO: Add IMDB dataset
# TODO: Check if DUE converges on other dataset than CLINC


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

    def to(self, device: Device) -> Dataset:
        self.batched_sequences = [
            (input_.to(device), target.to(device))
            for input_, target in self.batched_sequences
        ]

        return self

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
        add_eos_token: bool = True,
        is_sequence_classification: bool = False,
        **indexing_params: Dict[str, Any],
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
        add_eos_token: bool
            Indicate whether an <eos> token should be added. Default is True.
        is_sequence_classification: bool
            Add info on whether a dataset task is sequence classification. If so, the label will be extracted by
            splitting each line by the tab character and using the last split part as the label.
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
        self.add_eos_token = add_eos_token
        self.is_sequence_classification = is_sequence_classification
        self.label_encoder = LabelEncoder()

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

    def get_split(self, split: str) -> DataSplit:
        """
        Get a custom split of a dataset.

        Parameters
        ----------
        split: str
            Name of custom split.

        Returns
        -------
        DataSplit
            Return custom split.
        """
        # If train split hasn't been created yet, do that first to maintain consistent indexing
        if self._train is None:
            self.train

        if not hasattr(self, f"_{split}"):
            setattr(self, f"_{split}", self._load(split))

        return getattr(self, f"_{split}")

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
        split_paths = os.path.join(self.data_dir, self.name, self.splits[split])

        with codecs.open(split_paths, "rb", "utf-8") as file:
            if self.is_sequence_classification:
                sequences, sequence_labels = list(
                    zip(*[line.strip().split("\t") for line in file.readlines()])
                )

            else:
                sequences, sequence_labels = [
                    line.strip() for line in file.readlines()
                ], None

        # Initialize indexing
        if split == "train" and self.t2i is None:
            self.t2i = T2I.build(sequences, **self.indexing_params)

            if self.is_sequence_classification:
                self.label_encoder.fit(sequence_labels)

        return self._batch(split, sequences, sequence_labels)

    def _get_language_modelling_targets(
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

    def _batch(
        self,
        split: str,
        sequences: List[str],
        sequence_labels: Optional[List[str]] = None,
    ) -> DataSplit:
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
        sequence_labels: Optional[List[str]]
            List of labels for every sequence in the dataset. Only relevant if is_sequence_classification = True,
            otherwise None.

        Returns
        -------
        DataSplit
            DataSplit class containing indexed sequences and target labels.
        """
        batched_sequences = None

        # Pad sequences up to a certain length, with one single sequence per batch instance - this style is commonly
        # used for sequence prediction or seq2seq tasks.
        if self.batch_style == "padding":

            indexed_sequences = list(
                map(
                    torch.LongTensor,
                    self.t2i(
                        [
                            # Add <eos> if necessary before indexing
                            sequence
                            + (f" {self.t2i.eos_token}" if self.add_eos_token else "")
                            for sequence in sequences
                        ],
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

            sequence_labels = torch.LongTensor(
                self.label_encoder.transform(sequence_labels)
            )
            batched_labels = torch.split(
                sequence_labels, split_size_or_sections=self.batch_size
            )

        # Continuous batching style used for language modelling - sequences can continue over batch boundaries
        elif self.batch_style == "continuous":

            if self.is_sequence_classification:
                raise ValueError(
                    "Dataset cannot consist of continuous batches and be a sequence classification task."
                )

            indexed_sequences = list(
                map(
                    torch.LongTensor,
                    self.t2i(
                        [
                            # Add <eos> if necessary before indexing
                            sequence
                            + (f" {self.t2i.eos_token}" if self.add_eos_token else "")
                            for sequence in sequences
                        ]
                    ),
                )
            )

            if self.sequence_length is None:
                self.sequence_length = max(len(seq) for seq in indexed_sequences)

            # Filter out sequences which are too long
            indexed_sequences = list(
                filter(
                    lambda seq: len(seq) < self.sequence_length - 1, indexed_sequences
                )
            )

            indexed_sequences = torch.cat(indexed_sequences, dim=0)

            # Work out how cleanly we can divide the dataset into batch-sized parts
            num_batched_steps = indexed_sequences.shape[0] // self.batch_size

            # Trim off any extra elements that wouldn't cleanly fit (remainders)
            indexed_sequences = indexed_sequences.narrow(
                0, 0, num_batched_steps * self.batch_size
            )

            # Evenly divide the data across the bsz batches.
            raw_batches = indexed_sequences.view(self.batch_size, -1).t().contiguous()

            num_batches = math.ceil(num_batched_steps / (self.sequence_length + 1))
            batched_sequences = [
                raw_batches[
                    n
                    * (self.sequence_length + 1) : (n + 1)
                    * (self.sequence_length + 1),
                    :,
                ].t()
                for n in range(num_batches)
            ]

            batched_sequences, batched_labels = self._get_language_modelling_targets(
                split, batched_sequences
            )

        # Batches of (sequences, labels)
        return DataSplit(list(zip(batched_sequences, batched_labels)))


class LanguageModelingDataset(TextDataset, ABC):
    """
    A superclass for language modeling datasets.
    """

    def _get_language_modelling_targets(
        self,
        split: str,
        batched_sequences: BatchedSequences,
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

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        sequence_length: int,
        **indexing_params: Dict[str, Any],
    ):
        super().__init__(
            name="wikitext-103",
            data_dir=data_dir,
            splits={
                "train": "wiki.train.tokens",
                "valid": "wiki.valid.tokens",
                "test": "wiki.test.tokens",
            },
            batch_size=batch_size,
            batch_style="continuous",
            sequence_length=sequence_length,
            **indexing_params,
        )


class PennTreebankDataset(LanguageModelingDataset):
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


class SequenceClassificationDataset(TextDataset, ABC):
    """
    Superclass for sequence classification datasets.
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        splits: Dict[str, str],
        batch_size: int,
        sequence_length: int,
        **indexing_params: Dict[str, Any],
    ):
        super().__init__(
            name=name,
            data_dir=data_dir,
            splits=splits,
            batch_size=batch_size,
            batch_style="padding",
            sequence_length=sequence_length,
            is_sequence_classification=True,
            **indexing_params,
        )


class ClincDataset(SequenceClassificationDataset):
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
