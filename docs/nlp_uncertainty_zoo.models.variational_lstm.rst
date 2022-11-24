Variational LSTM
================

The variational LSTM is a Bayesian LSTM-variant that produces multiple different predictions by using different dropout masks.
Dropout `(Srivastava et al., 2014) <https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer,>`_
is a regulation technique that randomly sets connections between neurons in a neural network to zero during training in order to avoid co-adaption during training.
Importantly, this technique is disabled during inference.
In their work, `Gal & Ghahramani (2016a) <http://proceedings.mlr.press/v48/gal16.pdf>`_ propose to use Dropout during inference as well in order to approximate the weight posterior of neural networks.
In a follow-up work, `Gal & Ghahramani (2016b) <https://proceedings.neurips.cc/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf>`_ apply this technique to recurrent neural networks as well:
Most importantly, the technique is not just applied naively, but it is made sure that the same type of connection share the same dropout mask across time steps (see Figure 1 in the paper).

In the original work, the authors also use a specific kind of dropout for the embedding layer, which hurt performance and slowed down training when we replicated results, and which was conseqeuently omitted.
The application of MC Dropout to transformers can be found in :py:mod:`nlp_uncertainty_zoo.models.variational_transformer`.

Variational LSTM Module Documentation
=====================================

.. automodule:: nlp_uncertainty_zoo.models.variational_lstm
   :members:
   :show-inheritance:
   :undoc-members: