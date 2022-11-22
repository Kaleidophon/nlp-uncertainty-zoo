SNGP Transformer
================

The Spectral-normalized Gaussian Process transformer consists of a transformer-based feature extractor, on which then a Gaussian Process output layer is fitted.
This idea was proposed by `Liu et al. (2020) <https://arxiv.org/pdf/2205.00403.pdf>`_, and is an instance of
Deep Kernel learning `(Wilson et al., 2015) <https://arxiv.org/pdf/1511.02222.pdf>`_.
The spectral aspect is further explained in :py:mod:`nlp_uncertainty_zoo.models.spectral`, since the spectral normalization
is also shared by other models.

In this module, we implement two versions:

    * :py:class:`nlp_uncertainty_zoo.models.sngp_transformer.SNGPTransformer` / :py:class:`nlp_uncertainty_zoo.models.sngp_transformer.SNGPTransformerModule`: SNGP applied to a transformer trained from scratch. See :py:mod:`nlp_uncertainty_zoo.models.transformer` for more information on how to use the `Transformer` model & module.
    * :py:class:`nlp_uncertainty_zoo.models.sngp_transformer.SNGPBert` / :py:class:`nlp_uncertainty_zoo.models.sngp_transformer.SNGPBertModule`:  SNGP applied to pre-trained and then fine-tuned. See :py:mod:`nlp_uncertainty_zoo.models.bert` for more information on how to use the `Bert` model & module.

.. warning::
    The :py:class:`nlp_uncertainty_zoo.models.sngp_transformer.SNGPTransformer` / :py:class:`nlp_uncertainty_zoo.models.sngp_transformer.SNGPTransformerModule` were included for completeness, but might not be very stable.
    In `Ulmer et al. (2022) <https://arxiv.org/pdf/2210.15452.pdf>`_, it was already found that even with pre-trained BERT models as feature extractors, training was quite unstable, and would probably be even more unstable when training the underlying transformer from scratch.

All the important model logic is encapsulated in the :py:class:`nlp_uncertainty_zoo.models.sngp_transformer.SNGPModule` class in order to avoid code redundancies.
Since many NLP tasks involve many classes, we use the approximation detailed in `Appendix A.1 in the paper <https://arxiv.org/pdf/2205.00403.pdf>`_.
To be able to compute uncertainty metrics like :py:func:`nlp_uncertainty_zoo.utils.metrics.mutual_information`, we choose to **not** use the mean-field approximation of the posterior in equation (7).

SNGP Transformer Module Documentation
=====================================

.. automodule:: nlp_uncertainty_zoo.models.sngp_transformer
   :members:
   :show-inheritance:
   :undoc-members: