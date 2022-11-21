Model package
=============

This package contains all the model implementations in the repository, and some extra logic.
See the documentations for the individual models below for more information:

    * :py:mod:`nlp_uncertainty_zoo.models.bayesian_lstm`: Implementation of the Bayesian LSTM
    * :py:mod:`nlp_uncertainty_zoo.models.bert`: Implementation of a BERT wrapper class.
    * :py:mod:`nlp_uncertainty_zoo.models.ddu_transformer`: Implementation of a Deep Deterministic Uncertaity Transformer and BERT.
    * :py:mod:`nlp_uncertainty_zoo.models.dpp_transformer`: Implementation of a Determinantal Point-Process Transformer and BERT.
    * :py:mod:`nlp_uncertainty_zoo.models.lstm`: Implementation of Long-short term memory RNN.
    * :py:mod:`nlp_uncertainty_zoo.models.lstm_ensemble`: Implementation of an ensemble of LSTMs.
    * :py:mod:`nlp_uncertainty_zoo.models.model`: Explanation of the abstract Model and Module classes used in the repository.
    * :py:mod:`nlp_uncertainty_zoo.models.sngp_transformer`: Implementation of the Spectral-normalized Gaussian Process Transformer and BERT.
    * :py:mod:`nlp_uncertainty_zoo.models.spectral`: Implementation of superclasses using spectral normalization (used for :py:mod:`nlp_uncertainty_zoo.models.ddu_transformer` and :py:mod:`nlp_uncertainty_zoo.models.sngp_transformer`)
    * :py:mod:`nlp_uncertainty_zoo.models.st_tau_lstm`: Implementation of the ST-tau LSTM.
    * :py:mod:`nlp_uncertainty_zoo.models.transformer`: Implementation of a basic transformer.
    * :py:mod:`nlp_uncertainty_zoo.models.variational_lstm`: Implementation of a Variational LSTM using MC Dropout.
    * :py:mod:`nlp_uncertainty_zoo.models.variational_transformer`: Implementation of a Variational Transformer and BERT.
