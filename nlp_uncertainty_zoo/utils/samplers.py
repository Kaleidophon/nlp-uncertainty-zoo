"""
Sampler used to sub-sample different types of datasets.
"""

# STD
from abc import ABC, abstractmethod
from typing import Sized

# EXT
from torch.utils.data.sampler import Sampler


class Subsampler(Sampler, ABC):
    """
    Abstract base class of any sampler that sub-samples a dataset to a given target size.
    """

    def __init__(self, data_source: Sized, target_size: int):
        super().__init__(data_source)
        self.target_size = target_size
        self._analyze_data(data_source)

    @abstractmethod
    def _analyze_data(self, data_source: Sized):
        """
        Analyze the given data in order to determing the sampling strategy.

        Parameters
        ----------
        data_source: Sized
            DataLoader containing instances of data split.
        """
        pass


class LanguageModellingSampler(Subsampler):
    """
    Sampler specific to language modelling.  The sub-sampling strategy here is to approximately maintain the same
    distribution of sentence lengths as in the original corpus, and to maintain contiguous paragraphs of text spanning
    multiple sentences.
    """

    ...  # TODO


class SequenceClassificationSampler(Subsampler):
    """
    Sampler specific to sequence classification. The strategy here is to approximately maintain the same class
    distribution as in the original corpus, and to a lesser extent the same sequence length distribution.
    """

    ...  # TODO


class TokenClassificationSampler(Subsampler):
    """
    Sampler specific to sequence classification. The strategy here is to approximately maintain the same class
    distribution as in the original corpus, and to a lesser extent the same sequence length distribution. Compared to
    the SequenceClassificationSampler, all labels of a sequence are considered.
    """

    ...  # TODO
