"""
This module puts all the hyper-, training and preprocessing parameters used in this project into one single place.
"""

# PROJECT
from nlp_uncertainty_zoo.defaults import LANGUAGE_MODELLING_DEFAULT_PARAMS, SEQUENCE_CLASSIFICATION_DEFAULT_PARAMS
from nlp_uncertainty_zoo.models.variational_transformer import (
    VariationalTransformer,
    VariationalBert,
)
from nlp_uncertainty_zoo.models.variational_lstm import VariationalLSTM
from nlp_uncertainty_zoo.models.lstm_ensemble import LSTMEnsemble
from nlp_uncertainty_zoo.models.lstm import LSTM
from nlp_uncertainty_zoo.models.bayesian_lstm import BayesianLSTM
from nlp_uncertainty_zoo.models.st_tau_lstm import STTauLSTM
from nlp_uncertainty_zoo.models.transformer import Transformer
from nlp_uncertainty_zoo.models.ddu_transformer import DDUTransformer, DDUBert
from nlp_uncertainty_zoo.models.dpp_transformer import DPPBert, DPPTransformer
from nlp_uncertainty_zoo.models.sngp_transformer import SNGPTransformer, SNGPBert

# AVAILABLE MODELS
AVAILABLE_MODELS = {
    "lstm": LSTM,
    "lstm_ensemble": LSTMEnsemble,
    "variational_lstm": VariationalLSTM,
    "bayesian_lstm": BayesianLSTM,
    "st_tau_lstm": STTauLSTM,
    "transformer": Transformer,
    "variational_transformer": VariationalTransformer,
    "variational_bert": VariationalBert,
    "sngp_transformer": SNGPTransformer,
    "sngp_bert": SNGPBert,
    "ddu_transformer": DDUTransformer,
    "ddu_bert": DDUBert,
    "dpp_transformer": DPPTransformer,
    "dpp_bert": DPPBert,
}

DEFAULT_PARAMS = {
    "language_modelling": LANGUAGE_MODELLING_DEFAULT_PARAMS,
    "sequence_classification": SEQUENCE_CLASSIFICATION_DEFAULT_PARAMS
}