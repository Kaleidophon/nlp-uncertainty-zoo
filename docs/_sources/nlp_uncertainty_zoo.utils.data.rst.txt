Data
====

The contents of this module are mostly concerned with creating compatibility with the
`Huggingface transformers package <https://huggingface.co/docs/transformers/index>`_.
Specifically, the :py:class:`nlp_uncertainty_zoo.utils.data.DatasetBuilder` class tries to provide an easy interface to
load local datasets and utilizing the Huggingface code.
Furthermore, it supports the easy use of custom samplers defined in the :py:mod:`nlp_uncertainty_zoo.utils.samplers` module,
that produce representative sub-samples of training sets for different tasks.
These were for instance used in `Ulmer et al. (2022) <https://arxiv.org/pdf/2210.15452.pdf>`_ to show how the quality of uncertainty estimates behaves as a function of the available training data.

The following dataset builder classes are included:

    * :py:class:`nlp_uncertainty_zoo.utils.data.DatasetBuilder`: Abstract superclass that can be used for inheritance in order to support new task types.
    * :py:class:`nlp_uncertainty_zoo.utils.data.LanguageModellingDatasetBuilder`: Dataset builder used for language modelling, including both "classic" language modelling and masked language modelling. To indicate which type of language modelling us used, either `"next_token_prediction"` or `"mlm"` should be specified for the `type_` argument during initialization.
    * :py:class:`nlp_uncertainty_zoo.utils.data.ClassificationDatasetBuilder`: As the name suggests, this class is aimed at classification problems, both in terms of sequence labelling and sequence prediction. This is again specified by passing `"sequence_classification"` or `"token_classification"` in the `type_` argument during initialization. Dataset files are expected to be in the `.csv` format with tab-separated columns containing the sentence and label(s). When using sequence labelling, labels spanning multiple subword tokens will only be assigned to the first part, while the other subword tokens receive a `-100` label.

Furthermore, the module constain a modified version of Huggingface's `DataCollatorForLanguageModeling <:py:class:`nlp_uncertainty_zoo.utils.data.LanguageModellingDatasetBuilder`: >`_:
It seemed that for the classical next token prediction, the collator wouldn't produce the right offset between tokens and labels, i.e. the next tokens to be predicted.
The :py:class:`nlp_uncertainty_zoo.utils.data.ModifiedDataCollatorForLanguageModeling` provides a minimal modification of the original code to ensure this property.

Data Module Documentation
=========================

.. automodule:: nlp_uncertainty_zoo.utils.data
   :members:
   :show-inheritance:
   :undoc-members:
