Samplers
========

This module contains dataset sampler that are used with the :py:class:`nlp_uncertainty_zoo.utils.data.DatasetBuilder` class
to create representative sub-samples of training data.
These were for instance used in `Ulmer et al. (2022) <https://arxiv.org/pdf/2210.15452.pdf>`_ to show how the quality of uncertainty estimates behaves as a function of the available training data.

Right now, the module comprises three different samplers:

    * :py:class:`nlp_uncertainty_zoo.utils.samplers.LanguageModellingSampler`: Here, inputs are sub-sampled to primarily maintain the original distribution of sentence lengths like in the text. Also, multiple blocks of sentences are samples contiguously to maintain a notion of paragraphs.
    * :py:class:`nlp_uncertainty_zoo.utils.samplers.SequenceClassificationSampler`: Inputs are mostly sub-sampled to maintain the same class distributions as in the original corpus. Secondly, the same distribution of sequence lengths is also tried to be maintained.
    * :py:class:`nlp_uncertainty_zoo.utils.samplers.TokenClassificationSampler`: Same as the previous one. In order to maintain the same class distribution, the sequences are primarily sampled proportion to the cross-entropy between a sequence's label distribution and the global label distribution.

Samplers Module Documentation
=============================

.. automodule:: nlp_uncertainty_zoo.utils.samplers
   :members:
   :show-inheritance:
   :undoc-members:
