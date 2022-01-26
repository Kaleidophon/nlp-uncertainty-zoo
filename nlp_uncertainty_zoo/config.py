"""
This module puts all the hyper-, training and preprocessing parameters used in this project into one single place.
"""

# PROJECT
from nlp_uncertainty_zoo.data import PennTreebankBuilder, ClincBuilder, DanPlusBuilder
import nlp_uncertainty_zoo.model_configs as configs
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
from nlp_uncertainty_zoo.models.due_transformer import DUETransformer, DUEBert
from nlp_uncertainty_zoo.models.sngp_transformer import SNGPTransformer, SNGPBert

# AVAILABLE DATASETS AND MODELS
AVAILABLE_DATASETS = {
    "ptb": PennTreebankBuilder,
    "clinc": ClincBuilder,
    "dan+": DanPlusBuilder,
}
DATASET_TASKS = {
    "ptb": "language_modelling",
    "clinc": "sequence_classification",
    "dan+": "token_classification",
}
# TODO: Debug
AVAILABLE_MODELS = {
    # "lstm": LSTM,
    # "lstm_ensemble": LSTMEnsemble,
    # "variational_lstm": VariationalLSTM,
    # "bayesian_lstm": BayesianLSTM,
    # "st_tau_lstm": STTauLSTM,
    # "transformer": Transformer,
    # "variational_transformer": VariationalTransformer,
    # "variational_bert": VariationalBert,
    # "sngp_transformer": SNGPTransformer,
    "sngp_bert": SNGPBert,
    # "ddu_transformer": DDUTransformer,
    # "ddu_bert": DDUBert,
    # "due_transformer": DUETransformer,
    # "due_bert": DUEBert,
}

# MODEL HYPERPARAMETERS
# Hyperparameters by dataset and model
MODEL_PARAMS = {
    "ptb": configs.PTB_MODEL_PARAMS,
    "clinc": configs.CLINC_MODEL_PARAMS,
}
