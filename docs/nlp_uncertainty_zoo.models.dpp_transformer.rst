DPP Transformer
================

This module includes an implementation of the transformers using deep determinantal point processes by `Shelmanov et al. (2021) <https://aclanthology.org/2021.eacl-main.157.pdf>`_.
The idea is very related to using Monte Carlo dropout (see :py:mod:`nlp_uncertainty_zoo.models.variational_transformer`).
The difference is that dropout masks are constructed using correlation kernel between neurons, in order to obtain less correlated predictions overall.

In this model, we implement two versions:

    * :py:class:`nlp_uncertainty_zoo.models.dpp_transformer.DPPTransformer` / :py:class:`nlp_uncertainty_zoo.models.dpp_transformer.DPPTransformerModule`: DPPs applied to a transformer trained from scratch. See :py:mod:`nlp_uncertainty_zoo.models.transformer` for more information on how to use the `Transformer` model & module.
    * :py:class:`nlp_uncertainty_zoo.models.dpp_transformer.DPPBert` / :py:class:`nlp_uncertainty_zoo.models.dpp_transformer.DPPBertModule`:  DPPs applied to pre-trained and then fine-tuned. See :py:mod:`nlp_uncertainty_zoo.models.bert` for more information on how to use the `Bert` model & module.


DPP Transformer Module Documentation
====================================

.. automodule:: nlp_uncertainty_zoo.models.dpp_transformer
   :members:
   :show-inheritance:
   :undoc-members: