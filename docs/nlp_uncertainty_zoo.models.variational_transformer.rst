Variational Transformer
=======================

The variational transformer is a Bayesian variant that produces multiple different predictions by using different dropout masks.
Dropout `(Srivastava et al., 2014) <https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer,>`_
is a regulation technique that randomly sets connections between neurons in a neural network to zero during training in order to avoid co-adaption during training.
Importantly, this technique is disabled during inference.
In their work, `Gal & Ghahramani (2016a) <http://proceedings.mlr.press/v48/gal16.pdf>`_ propose to use Dropout during inference as well in order to approximate the weight posterior of neural networks.
In a follow-up work, `Gal & Ghahramani (2016b) <https://proceedings.neurips.cc/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf>`_ apply this technique to recurrent neural networks as well, and
in `Xiao et al., (2020) <http://arxiv.org/abs/2006.08344>`_ to transformer architectures.

.. warning::
    In `Xiao et al., (2020) <http://arxiv.org/abs/2006.08344>`_, it is not fully specified if MC Dropout is used with all available dropout layers.
    We opted for this approach, and found encouraging results `(Ulmer et al., 2022) <https://arxiv.org/pdf/2210.15452.pdf>`_ .

In this module, we implement two versions:

    * :py:class:`nlp_uncertainty_zoo.models.variational_transformer.VariationalTransformer` / :py:class:`nlp_uncertainty_zoo.models.variational_transformer.VariationalTransformerModule`: MC Dropout applied to a transformer trained from scratch. See :py:mod:`nlp_uncertainty_zoo.models.transformer` for more information on how to use the `Transformer` model & module.
    * :py:class:`nlp_uncertainty_zoo.models.variational_transformer.VariationalBert` / :py:class:`nlp_uncertainty_zoo.models.variational_transformer.VariationalBertModule`: MC Dropout applied to pre-trained and then fine-tuned. See :py:mod:`nlp_uncertainty_zoo.models.bert` for more information on how to use the `Bert` model & module.

The application of MC Dropout to LSTMs can be found in :py:mod:`nlp_uncertainty_zoo.models.variational_lstm`.

Variational Transformer Module Documentation
=====================================

.. automodule:: nlp_uncertainty_zoo.models.variational_transformer
   :members:
   :show-inheritance:
   :undoc-members: