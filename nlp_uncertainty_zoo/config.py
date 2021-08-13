"""
This module puts all the hyper-, training and preprocessing parameters used in this project into one single place.
"""

# TODO: Add tokenizers

# EXT
from scipy.stats import uniform

# PROJECT
from nlp_uncertainty_zoo.models.composer import Composer
from nlp_uncertainty_zoo.datasets import (
    Wikitext103Dataset,
    PennTreebankDataset,
    ClincDataset,
)
import nlp_uncertainty_zoo.dataset_configs as configs
from nlp_uncertainty_zoo.models.variational_transformer import VariationalTransformer
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

# TRAINING PARAMETERS
# Training parameters by dataset and model
TRAIN_PARAMS = {
    "wikitext-103": configs.WIKITEXT_TRAIN_PARAMS,
    "ptb": configs.PTB_TRAIN_PARAMS,
    "clinc": configs.CLINC_TRAIN_PARAMS,
}


# MODEL HYPERPARAMETERS
# Hyperparameters by dataset and model
SHARED_MODEL_PARAMS = {
    "wikitext-103": configs.WIKITEXT_SHARED_MODEL_PARAMS,
    "ptb": configs.PTB_MODEL_PARAMS,
    "clinc": configs.CLINC_SHARED_MODEL_PARAMS,
}
_MODEL_PARAMS = {
    "wikitext-103": configs.WIKITEXT_MODEL_PARAMS,
    "ptb": configs.PTB_MODEL_PARAMS,
    "clinc": configs.CLINC_MODEL_PARAMS,
}
MODEL_PARAMS = {
    dataset: {
        model_name: {**SHARED_MODEL_PARAMS[dataset], **model_params}
        for model_name, model_params in model_dicts.items()
    }
    for dataset, model_dicts in _MODEL_PARAMS.items()
}

# TODO: Move this into dataset_configs modules
# HYPERPARAMETER SEARCH
# Number of tested configurations per dataset per model
NUM_EVALS = {
    "wikitext-103": {
        "lstm": 2,
        "variational_lstm": 20,
        "transformer": 2,
        "variational_transformer": 10,
        "sngp_transformer": 10,
        "ddu_transformer": 10,
    }
}

# Search ranges / options per dataset per model
PARAM_SEARCH = {
    "wikitext-103": {
        "lstm": {"num_layers": list(range(2, 6)), "dropout": uniform(0.1, 0.4)},
        "variational_lstm": {},
        "transformer": {
            "num_layers": list(range(2, 6)),
            "dropout": uniform(0.1, 0.4),
            "num_heads": [5, 10, 15],
        },
        "variational_transformer": {},
        "sngp_transformer": {},
        "ddu_transformer": {},
    }
}
