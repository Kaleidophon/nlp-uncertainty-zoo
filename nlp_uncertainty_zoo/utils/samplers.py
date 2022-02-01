"""
Sampler used to sub-sample different types of datasets.
"""

# STD
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from typing import Sized, Dict, Optional, Tuple

# EXT
from torch.utils.data.sampler import Sampler


def _create_probs_from_dict(freq_dict: Dict[int, int]) -> np.array:
    """
    Auxiliary function creating a numpy array containing a categorical distribution over integers from
    a dictionary of frequencies.
    """
    probs = np.zeros(max(freq_dict.keys()) + 1)

    for key, freq in freq_dict.items():
        probs[key] = freq

    probs /= sum(probs)

    return probs


class Subsampler(Sampler, ABC):
    """
    Abstract base class of any sampler that sub-samples a dataset to a given target size.
    """

    def __init__(
        self, data_source: Sized, target_size: int, seed: Optional[int] = None
    ):
        super().__init__(data_source)
        self.target_size = target_size
        self.seed = seed
        self.indices = None
        self._create_indices(data_source)

    @abstractmethod
    def _create_indices(self, data_source: Sized):
        """
        Analyze the given data in order to determing the sampling strategy.

        Parameters
        ----------
        data_source: Sized
            Data containing instances of data split.
        """
        pass

    def __iter__(self):
        return iter(self.indices)


class LanguageModellingSampler(Subsampler):
    """
    Sampler specific to language modelling.  The sub-sampling strategy here is to approximately maintain the same
    distribution of sentence lengths as in the original corpus, and to maintain contiguous paragraphs of text spanning
    multiple sentences.
    """

    def __init__(
        self,
        data_source: Sized,
        target_size: int,
        sample_range: Tuple[int, int],
        seed: Optional[int] = None,
    ):
        self.length2instances = defaultdict(lambda: [])
        self.sample_range = sample_range
        self.seq_lengths = defaultdict(int)
        super().__init__(data_source, target_size, seed)

    def _create_indices(self, data_source: Sized):
        """
        Analyze the given data in order to determing the sampling strategy.

        Parameters
        ----------
        data_source: Sized
            Data containing instances of data split.
        """

        # Go through data and categorize it
        for i, instance in enumerate(data_source):
            seq_length = len(instance["input_ids"])
            self.seq_lengths[seq_length] += 1
            self.length2instances[seq_length].append(i)

        # Compute probability of sampling an instance based on class and sentence length
        instance_probs = np.zeros(len(data_source))
        seq_length_probs = _create_probs_from_dict(self.seq_lengths)

        for seq_length in self.length2instances:

            for i in self.length2instances[seq_length]:
                # Probability for an instance to be sampled is the probability of the class label times the
                # probability of the sequence length given the class label divided by the number of sequences
                # with that same class and sequence length
                instance_probs[i] = seq_length_probs[seq_length] / len(
                    self.length2instances[seq_length]
                )

        if self.seed is not None:
            np.random.seed(self.seed)

        self.indices = []

        while len(self.indices) < self.target_size:
            # Pick index
            index = np.random.choice(
                np.arange(len(data_source)),
                size=1,
                replace=False,
                p=instance_probs,
            )[0]

            # Pick length
            length = np.random.choice(np.arange(*self.sample_range), size=1)[0]
            offset = min(
                len(data_source) - 1, length, self.target_size - len(self.indices)
            )

            # Add the sampled indices
            self.indices.extend(range(index, index + offset))

            # Mask out the sampled indices and re-normalize the distribution
            instance_probs[index : index + offset] = 0
            instance_probs /= sum(instance_probs)


class SequenceClassificationSampler(Subsampler):
    """
    Sampler specific to sequence classification. The strategy here is to approximately maintain the same class
    distribution as in the original corpus, and to a lesser extent the same sequence length distribution.
    """

    def __init__(
        self, data_source: Sized, target_size: int, seed: Optional[int] = None
    ):
        self.class2length2instances = defaultdict(lambda: defaultdict(lambda: []))
        self.classes = defaultdict(int)
        self.class_lengths = defaultdict(lambda: defaultdict(int))
        super().__init__(data_source, target_size, seed)

    def _create_indices(self, data_source: Sized):
        """
        Analyze the given data in order to determing the sampling strategy.

        Parameters
        ----------
        data_source: Sized
            Data containing instances of data split.
        """

        # Go through data and categorize it
        for i, instance in enumerate(data_source):
            label = instance["labels"].item()
            seq_length = len(instance["input_ids"])
            self.class2length2instances[label][seq_length].append(i)
            self.classes[label] += 1
            self.class_lengths[label][seq_length] += 1

        # Compute probability of sampling an instance based on class and sentence length
        label_probs = _create_probs_from_dict(self.classes)
        instance_probs = np.zeros(len(data_source))

        for label in self.classes:
            seq_length_probs = _create_probs_from_dict(self.class_lengths[label])

            for seq_length in self.class2length2instances[label]:
                for i in self.class2length2instances[label][seq_length]:
                    # Probability for an instance to be sampled is the probability of the class label times the
                    # probability of the sequence length given the class label divided by the number of sequences
                    # with that same class and sequence length
                    instance_probs[i] = (
                        label_probs[label]
                        * seq_length_probs[seq_length]
                        / len(self.class2length2instances[label][seq_length])
                    )

        if self.seed is not None:
            np.random.seed(self.seed)

        self.indices = np.random.choice(
            np.arange(len(data_source)),
            size=self.target_size,
            replace=False,
            p=instance_probs,
        ).tolist()


class TokenClassificationSampler(Subsampler):
    """
    Sampler specific to sequence classification. The strategy here is to approximately maintain the same class
    distribution as in the original corpus, and to a lesser extent the same sequence length distribution. Compared to
    the SequenceClassificationSampler, all labels of a sequence are considered.
    """

    ...  # TODO: Implement _analyse_data
