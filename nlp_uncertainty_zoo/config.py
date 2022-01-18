"""
This module puts all the hyper-, training and preprocessing parameters used in this project into one single place.
"""

# PROJECT
from nlp_uncertainty_zoo.data import (
    Wikitext103Dataset,
    PennTreebankDataset,
    ClincDataset,
)
import nlp_uncertainty_zoo.dataset_configs as configs
from nlp_uncertainty_zoo.models.variational_transformer import (
    VariationalTransformer,
    VariationalBert,
)
from nlp_uncertainty_zoo import (
    VariationalLSTM,
    LSTMEnsemble,
    SNGPTransformer,
    DUETransformer,
    DDUTransformer,
)
from nlp_uncertainty_zoo.models.lstm import LSTM
from nlp_uncertainty_zoo.models.bayesian_lstm import BayesianLSTM
from nlp_uncertainty_zoo.models.st_tau_lstm import STTauLSTM
from nlp_uncertainty_zoo.models.transformer import Transformer

# AVAILABLE DATASETS AND MODELS
AVAILABLE_DATASETS = {
    "wikitext-103": Wikitext103Dataset,
    "ptb": PennTreebankDataset,
    "clinc": ClincDataset,
}
AVAILABLE_MODELS = {
    # "composer": Composer,
    "lstm": LSTM,
    "lstm_ensemble": LSTMEnsemble,
    "variational_lstm": VariationalLSTM,
    "bayesian_lstm": BayesianLSTM,
    "st_tau_lstm": STTauLSTM,
    "transformer": Transformer,
    "variational_transformer": VariationalTransformer,
    "variational_bert": VariationalBert,
    "sngp_transformer": SNGPTransformer,
    "ddu_transformer": DDUTransformer,
    "due_transformer": DUETransformer,
}

# PREPROCESSING PARAMETERS
# List of preprocessing parameters by dataset

PREPROCESSING_PARAMS = {
    "wikitext-103": configs.WIKITEXT_PREPROCESSING_PARAMS,
    "ptb": configs.PTB_PREPROCESSING_PARAMS,
    "clinc": configs.CLINC_PREPROCESSING_PARAMS,
}

# MODEL HYPERPARAMETERS
# Hyperparameters by dataset and model
MODEL_PARAMS = {
    "wikitext-103": configs.WIKITEXT_MODEL_PARAMS,
    "ptb": configs.PTB_MODEL_PARAMS,
    "clinc": configs.CLINC_MODEL_PARAMS,
}
