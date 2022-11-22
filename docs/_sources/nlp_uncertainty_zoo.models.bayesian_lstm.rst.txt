Bayesian LSTM
=============

The Bayesian LSTM is based on the concept of Bayes-by-backprop introduced by `Blundell et al. (2015) <http://proceedings.mlr.press/v37/blundell15.pdf>`_, applied to recurrent networks by `Fortunato et al. (2017) <https://arxiv.org/pdf/1704.02798.pdf>`_.
The idea is that instead of learning one single value per parameter, we learn a normal distribution over parameter values (thus, we actually learn *two* parameters, the mean and variance of every network parameter).
During inference, we sample one parameter set from these distributions to make a prediction.

In this case, we implement the Bayesian LSTM using the `Blitz <https://github.com/piEsposito/blitz-bayesian-deep-learning>`_ package.

Bayesian LSTM Module Documentation
============================

.. automodule:: nlp_uncertainty_zoo.models.bayesian_lstm
   :members:
   :show-inheritance:
   :undoc-members: