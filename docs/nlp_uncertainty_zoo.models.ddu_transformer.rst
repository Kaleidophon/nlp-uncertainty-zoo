DDU Transformer
================

This module includes an implementation of the Deep Deterministic Uncertainty (DDU) transformer by `Mukhoti et al. (2021) <http://arxiv.org/abs/2102.11582>`_.
The approach involves fine-tuning a transformer, and then fitting a Gaussian Mixture Model (GMM) on its activations.
Model uncertainty is then computed based on the log probability of an input based on its encoding under the GMM.

In this model, we implement two versions:

    * :py:class:`nlp_uncertainty_zoo.models.ddu_transformer.DDUTransformer` / :py:class:`nlp_uncertainty_zoo.models.ddu_transformer.DDUTransformerModule`: DDU applied to a transformer trained from scratch. See :py:mod:`nlp_uncertainty_zoo.models.transformer` for more information on how to use the `Transformer` model & module.
    * :py:class:`nlp_uncertainty_zoo.models.ddu_transformer.DDUBert` / :py:class:`nlp_uncertainty_zoo.models.ddu_transformer.DDUBertModule`:  DDU applied to pre-trained and then fine-tuned. See :py:mod:`nlp_uncertainty_zoo.models.bert` for more information on how to use the `Bert` model & module.

Internally, the Gaussian mixture model is fit in the :py:meth:`nlp_uncertainty_zoo.models.ddu_transformer.DDUBert._finetune()` and :py:meth:`nlp_uncertainty_zoo.models.ddu_transformer.DDUTransformer._finetune()` method, which should be called using the data's validation split.

All the important model info is also encapsulated in the :py:class:`nlp_uncertainty_zoo.models.ddu_transformer.DDUMixin` class in order to avoid code redundancies.

DDU Transformer Module Documentation
====================================

.. automodule:: nlp_uncertainty_zoo.models.ddu_transformer
   :imported-members:
   :members:
   :show-inheritance:
   :undoc-members: