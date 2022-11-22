LSTM Ensemble
=============

Implementation of an ensemble of :py:mod:`nlp_uncertainty_zoo.models.lstm` instances.
This is inspired by the work of `Lakshminarayanan et al. (2017) <https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf>`_, showing that an ensemble is a strong model for generalization and uncertainty quantification due to its diversity.
This was also confirmed in the work of `Ovadia et al. (2019) <https://arxiv.org/pdf/1906.02530.pdf>`_ in a computer vision setting, and in `Ulmer et al. (2022) <https://arxiv.org/pdf/2210.15452.pdf>`_ for natural language processing, where ensembles for LSTMs performed en par or better than pre-trained BERT models.

LSTM Ensemble Module Documentation
==================================

.. automodule:: nlp_uncertainty_zoo.models.lstm_ensemble
   :members:
   :show-inheritance:
   :undoc-members: