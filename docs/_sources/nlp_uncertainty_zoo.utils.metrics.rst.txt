Uncertainty metrics
===================

This module constains all general-purpose uncertainty metrics used by models in the repository.
An uncertainty metric quantified the uncertainty of a model in its prediction, where higher values of the metric indicate a higher uncertainty.
Uncertainty metrics take the unnormalized logits of a model as an input.
Some metrics might not be included here, since they are model-specified, see for instance :py:meth:`nlp_uncertainty_zoo.models.ddu_transformer.DDUMixin.gmm_predict`, which
becomes the `log_prob` metric for :py:class:`nlp_uncertainty_zoo.models.ddu_transformer.DDUTransformer` and :py:class:`nlp_uncertainty_zoo.models.ddu_transformer.DDUBert`.

Metric Module Documentation
===========================

.. automodule:: nlp_uncertainty_zoo.utils.metrics
   :members:
   :show-inheritance:
   :undoc-members:
