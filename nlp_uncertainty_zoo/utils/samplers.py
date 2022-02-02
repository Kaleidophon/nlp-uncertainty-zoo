"""
Sampler used to sub-sample different types of datasets. In each class, some statistics about the distribution of inputs
is built, and then indices of instances from the dataset are subs-ampled based on these statistics.
"""

# STD
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import numpy as np
from typing import Sized, Dict, Optional, Tuple

# EXT
from torch.utils.data.sampler import Sampler

# TODO: Use joblib to speed up sampling?


def create_probs_from_dict(
    freq_dict: Dict[int, int], max_label: Optional[int] = None
) -> np.array:
    """
    Auxiliary function creating a numpy array containing a categorical distribution over integers from
    a dictionary of frequencies.

    Parameters
    ----------
    freq_dict: Dict[int, int]
        Dictionary mapping from class labels to frequencies.
    max_label: Optional[int]
        Maximum value of a class label aka number of classes (minus 1). If None, tyis is based on the maximum valued
        key in freq_dict.

    Returns
    -------
    np.array
        Distribution over class labels as a numpy array.
    """
    if max_label is None:
        max_label = max(freq_dict.keys())

    probs = np.zeros(max_label + 1)

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
        """
        Initialize a sub-sampler.

        Parameters
        ----------
        data_source: Sized
            Iterable of data corresponding to a split. Usually is a list of dicts, containing input ids, attention masks
            and labels for each instance.
        target_size: int
            Number of instances that should be contained in the sub-sampled data set.
        seed: Optional[int]
            Seed set for reproducibility.
        """
        super().__init__(data_source)
        self.target_size = target_size
        self.seed = seed
        self.indices = None
        self._create_indices(data_source)

    @abstractmethod
    def _create_indices(self, data_source: Sized):
        """
        Analyze the given data in order to determining the sampling strategy.

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
    Sampler specific to language modelling. The sub-sampling strategy here is to approximately maintain the same
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
        """
        Initialize a sub-sampler for language modelling data sets.

        Parameters
        ----------
        data_source: Sized
            Iterable of data corresponding to a split. Usually is a list of dicts, containing input ids, attention masks
            and labels for each instance.
        target_size: int
            Number of instances that should be contained in the sub-sampled data set.
        sample_range: Tuple[int, int]
            Length of paragraphs that are sampled from the corpus. sample_range determines the ranges from which the
            length is sampled uniformly.
        seed: Optional[int]
            Seed set for reproducibility.
        """
        self.length2instances = defaultdict(
            lambda: []
        )  # List of instances with the same sentence length
        self.sample_range = sample_range
        self.seq_length_freqs = defaultdict(int)  # Frequency of sentence lengths
        super().__init__(data_source, target_size, seed)

    def _create_indices(self, data_source: Sized):
        """
        Analyze the given data in order to determining the sampling strategy.

        Parameters
        ----------
        data_source: Sized
            Data containing instances of data split.
        """

        # Go through data and categorize it
        for i, instance in enumerate(data_source):
            seq_length = len(instance["input_ids"])
            self.seq_length_freqs[seq_length] += 1
            self.length2instances[seq_length].append(i)

        # Compute probability of sampling an instance based on class and sentence length
        instance_probs = np.zeros(len(data_source))
        seq_length_probs = create_probs_from_dict(self.seq_length_freqs)

        for seq_length in self.length2instances:

            for i in self.length2instances[seq_length]:
                # Probability for an instance to be sampled is the probability of the sequence length divided by the
                # number of sequences with that same length
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
            )  # Sampled paragraph should not flow over corpus boundaries or create more instances than specified

            # Add the sampled indices
            self.indices.extend(range(index, index + offset))

            # Mask out the sampled indices so they cannot be sampled again and re-normalize the distribution
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
        """
        Initialize a sub-sampler for sequence classification tasks.

        Parameters
        ----------
        data_source: Sized
            Iterable of data corresponding to a split. Usually is a list of dicts, containing input ids, attention masks
            and labels for each instance.
        target_size: int
            Number of instances that should be contained in the sub-sampled data set.
        seed: Optional[int]
            Seed set for reproducibility.
        """
        # Dictionary mapping to a list instances of the same class and same sentence lengths
        self.class2length2instances = defaultdict(lambda: defaultdict(lambda: []))
        self.class_freqs = defaultdict(int)  # Frequency of classes
        # Frequencies of sentence lengths of the same class
        self.class_length_freqs = defaultdict(lambda: defaultdict(int))
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
            self.class_freqs[label] += 1
            self.class_length_freqs[label][seq_length] += 1

        # Compute probability of sampling an instance based on class and sentence length
        label_probs = create_probs_from_dict(self.class_freqs)
        instance_probs = np.zeros(len(data_source))

        for label in self.class_freqs:
            seq_length_probs = create_probs_from_dict(self.class_length_freqs[label])

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

    def __init__(
        self,
        data_source: Sized,
        target_size: int,
        ignore_label: int = -100,
        seed: Optional[int] = None,
    ):
        """
        Initialize a sub-sampler for token classification tasks.

        Parameters
        ----------
        data_source: Sized
            Iterable of data corresponding to a split. Usually is a list of dicts, containing input ids, attention masks
            and labels for each instance.
        target_size: int
            Number of instances that should be contained in the sub-sampled data set.
        ignore_label: int
            Determine label that should be ignored when computing input statistics used for sampling. Default is -100.
        seed: Optional[int]
            Seed set for reproducibility.
        """
        self.length2instances = defaultdict(
            lambda: []
        )  # List of instances with the same sentence length
        self.seq_length_freqs = defaultdict(int)  # Frequency of sentence lengths
        self.label_freqs = {}  # Frequencies of labels (over the whole corpus)
        self.instance2label_freqs = {}  # Frequencies of labels (per instance)
        self.ignore_label = ignore_label
        super().__init__(data_source, target_size, seed)

    def _create_indices(self, data_source: Sized):
        """
        Analyze the given data in order to determining the sampling strategy.

        Parameters
        ----------
        data_source: Sized
            Data containing instances of data split.
        """

        # Go through data and categorize it
        for i, instance in enumerate(data_source):
            seq_length = len(instance["input_ids"])
            label_freqs = Counter(instance["labels"].tolist())
            del label_freqs[self.ignore_label]
            self.seq_length_freqs[seq_length] += 1
            self.length2instances[seq_length].append(i)
            self.label_freqs.update(
                label_freqs
            )  # Update label distribution for whole dataset
            self.instance2label_freqs[
                i
            ] = label_freqs  # Update label freqs for instance

        # Compute probability of sampling an instance based on class and sentence length
        instance_probs = np.zeros(len(data_source))
        seq_length_probs = create_probs_from_dict(self.seq_length_freqs)
        label_probs = create_probs_from_dict(self.label_freqs)

        for seq_length in self.length2instances:

            log_lik = lambda p, q: sum(
                p * np.log(q)
            )  # Expected log-likelihood under corpus label distribution
            # Create a probability distribution over instances of the sample length by computing the log-likelihood
            # between the label distribution of an instance and the label distribution of a corpus and normalizing
            # all the scores for one length "bucket" into the 0, 1 range. This way, sampling instances that match the
            # overall label distribution becomes more likely.

            # First, create distributions over labels per instances
            seq_length_instance_probs = [
                create_probs_from_dict(
                    self.instance2label_freqs[i], max_label=len(label_probs) - 1
                )
                for i in self.length2instances[seq_length]
            ]
            # Secondly, compute cross-entropy scores
            seq_length_instance_probs = np.array(
                [
                    log_lik(label_probs, label_dist + 1e-5)
                    for label_dist in seq_length_instance_probs
                ]
            )
            seq_length_instance_probs += min(
                seq_length_instance_probs
            )  # Make all values positive
            seq_length_instance_probs /= sum(seq_length_instance_probs)  # Normalize

            for i, instance_prob in zip(
                self.length2instances[seq_length], seq_length_instance_probs
            ):
                # Probability for an instance to be sampled is the probability of the sequence length times the
                # probability created by checking the frequencies of labels in a sequence with the overall labels in
                # the corpus
                instance_probs[i] = seq_length_probs[seq_length] * instance_prob

        if self.seed is not None:
            np.random.seed(self.seed)

        self.indices = np.random.choice(
            np.arange(len(data_source)),
            size=self.target_size,
            replace=False,
            p=instance_probs,
        ).tolist()
