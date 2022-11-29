Spectral
========

This module contains transformer implementations using spectral normalization, used for instance by :py:mod:`nlp_uncertainty_zoo.models.ddu_transformer` or :py:mod:`nlp_uncertainty_zoo.models.sngp_transformer`.
The main idea is that the functions learned by normal network should fulfill a bi-Lipschitz constraint (see `Mukhoti et al., 2021<http://arxiv.org/abs/2102.11582>`_):

    * On the one hand, models should smooth in order to avoid for instance adversarial attacks and promote robustness
    * On the other hand, models should still be sensitive enough to detect out-of-distribution examples.

In networks using residual connections (as contained inside the transformer architecture), this condition can be approximately fulfilled using spectral normalization, where the spectral norm (the largest singular value) of weight matrices is upper-bounded by some constant (chosen by the users).
In practice, the spectral normalization is applied as a hook to model parameters after every training step.

In this module, we implement two versions:

    * :py:class:`nlp_uncertainty_zoo.models.spectral.SpectralTransformerModule`: Spectral normalization applied to a transformer trained from scratch. See :py:mod:`nlp_uncertainty_zoo.models.transformer` for more information on how to use the `Transformer` model & module.
    * :py:class:`nlp_uncertainty_zoo.models.spectral.SpectralBertModule`:  Spectral normalization applied to pre-trained and then fine-tuned. See :py:mod:`nlp_uncertainty_zoo.models.bert` for more information on how to use the `Bert` model & module.

The main logic is contained in :py:class:`nlp_uncertainty_zoo.models.spectral.SpectralNormFC` and :py:func:`nlp_uncertainty_zoo.models.spectral.spectral_norm_fc` which were copied
verbatim from `the repository of Joost van Amsterfoort <https://github.com/y0ast/DUE>`_ due to import issues.

Spectral Module Documentation
=============================

.. automodule:: nlp_uncertainty_zoo.models.spectral
   :members:
   :show-inheritance:
   :undoc-members: