"""
This module puts all the hyper-, training and preprocessing parameters used in this project into one single place.
"""

# EXT
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
import torch

# PROJECT
from src.datasets import Wikitext103Dataset, PennTreebankDataset
from src.dropout import VariationalLSTM, VariationalTransformer
from src.lstm import LSTM
from src.spectral import SNGPTransformer, DDUTransformer
from src.transformer import Transformer

# AVAILABLE DATASETS AND MODELS
AVAILABLE_DATASETS = {"wikitext-103": Wikitext103Dataset, "ptb": PennTreebankDataset}
AVAILABLE_MODELS = {
    "lstm": LSTM,
    "variational_lstm": VariationalLSTM,
    "transformer": Transformer,
    "variational_transformer": VariationalTransformer,
    "sngp_transformer": SNGPTransformer,
    "ddu_transformer": DDUTransformer,
}

# PREPROCESSING PARAMETERS
# List of preprocessing parameters by dataset

SHARED_PREPROCESSING_PARAMS = {}
_PREPROCESSING_PARAMS = {
    "wikitext-103": {"batch_size": 64, "sequence_length": 30, "min_freq": 20},
    "ptb": {
        "batch_size": 20,
        "sequence_length": 35,
        "max_size": 9999,
    },  # - <unk> token
    "clinc": {"sequence_length": 32},
}

# Update shared preprocessing params wth dataset-specific params
PREPROCESSING_PARAMS = {
    dataset: {**SHARED_PREPROCESSING_PARAMS, **preprocessing_params}
    for dataset, preprocessing_params in _PREPROCESSING_PARAMS.items()
}

# TRAINING PARAMETERS
# Training parameters by dataset and model
SHARED_TRAIN_PARAMS = {
    "wikitext-103": {"num_epochs": 1, "gamma": 1},
    "ptb": {"gamma": 1},
    "clinc": {},
}
_TRAIN_PARAMS = {
    "wikitext-103": {
        "lstm": {"lr": 0.01},
        "variational_lstm": {"lr": 0.01},
        "transformer": {"lr": 0.01, "gamma": 0.95},
        "variational_transformer": {"lr": 0.01, "gamma": 0.95, "weight_decay": 0},
        "sngp_transformer": {"lr": 0.2, "gamma": 0.6, "weight_decay": 0.01},
        "ddu_transformer": {"lr": 0.01, "gamma": 0.95},
    },
    "ptb": {
        # Taken from  https://github.com/yaringal/BayesianRNN/blob/master/LM_code/main_new_dropout_SOTA.lua
        "variational_lstm": {
            "early_stopping": True,
            "weight_decay": 1e-7,
            "lr": 1,
            "num_epochs": 55,
            # "early_stopping_pat": 10,
            "grad_clip": 10,
            "gamma": 0.74,  # 1 / 1.35; in the Gal implementation you divide by gamma
            "milestones": torch.LongTensor(range(13, 54, 1)),
            "init_weight": 0.04,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        }
    },
    "clinc": {
        "sngp_transformer": {
            "lr": 5e-5,
            "length_scale": 2,
            "weight_decay": 0.1,
            "num_epochs": 40,
        }
    },
}
TRAIN_PARAMS = {
    dataset: {
        model_name: {**SHARED_TRAIN_PARAMS[dataset], **model_train_params}
        for model_name, model_train_params in model_train_dicts.items()
    }
    for dataset, model_train_dicts in _TRAIN_PARAMS.items()
}


# MODEL HYPERPARAMETERS
# Hyperparameters by dataset and model
SHARED_MODEL_PARAMS = {
    "wikitext-103": {
        "input_size": 100,
        "hidden_size": 100,
        "output_size": 20245,
        "vocab_size": 20245,
    },
    "ptb": {},
    "clinc": {},
}
_MODEL_PARAMS = {
    "wikitext-103": {
        "lstm": {"num_layers": 2, "dropout": 0.2},
        "variational_lstm": {"num_layers": 2, "dropout": 0.35},
        "transformer": {
            "num_layers": 6,
            "input_dropout": 0.2,
            "dropout": 0.2,
            "num_heads": 5,
            "sequence_length": 30,
        },
        "variational_transformer": {
            "num_layers": 6,
            "input_dropout": 0.2,
            "dropout": 0.2,
            "num_heads": 5,
            "sequence_length": 30,
            "num_predictions": 100,
        },
        "sngp_transformer": {
            "num_layers": 6,
            "input_dropout": 0.2,
            "dropout": 0.2,
            "num_heads": 5,
            "sequence_length": 30,
        },
        "ddu_transformer": {
            "num_layers": 6,
            "input_dropout": 0.2,
            "dropout": 0.2,
            "num_heads": 5,
            "sequence_length": 30,
        },
    },
    "ptb": {
        # Taken from https://github.com/yaringal/BayesianRNN/blob/master/LM_code/main_new_dropout_SOTA.lua
        "variational_lstm": {
            "num_layers": 2,
            "hidden_size": 1500,
            "input_size": 1500,
            "embedding_dropout": 0.3,  # dropout_x, Large model Gal & Ghrahramani (2016)
            "layer_dropout": 0.5,  # dropout_i / dropout_o, Large model Gal & Ghrahramani (2016)
            "time_dropout": 0.3,  # dropout_h, Large model Gal & Ghrahramani (2016)
            "vocab_size": 10000,
            "output_size": 10000,
            "num_predictions": 100,
        }
    },
    "clinc": {
        "sngp_transformer": {
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "vocab_size": 10000,
        }
    },
}
MODEL_PARAMS = {
    dataset: {
        model_name: {**SHARED_MODEL_PARAMS[dataset], **model_params}
        for model_name, model_params in model_dicts.items()
    }
    for dataset, model_dicts in _MODEL_PARAMS.items()
}

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
