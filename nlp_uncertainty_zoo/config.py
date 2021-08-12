"""
This module puts all the hyper-, training and preprocessing parameters used in this project into one single place.
"""

# TODO: Add tokenizers

# EXT
from scipy.stats import uniform
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import transformers

# PROJECT
from nlp_uncertainty_zoo.models.composer import Composer
from nlp_uncertainty_zoo.datasets import (
    Wikitext103Dataset,
    PennTreebankDataset,
    ClincDataset,
)
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
    "composer": Composer,
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

SHARED_PREPROCESSING_PARAMS = {}
_PREPROCESSING_PARAMS = {
    "wikitext-103": {"batch_size": 64, "sequence_length": 30, "min_freq": 20},
    "ptb": {
        "batch_size": 20,
        "sequence_length": 35,
        # "max_size": 10000,  # PTB has exactly 10000 types
    },
    "clinc": {"batch_size": 32, "sequence_length": 32},
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
        "lstm": {
            "early_stopping": True,
            "weight_decay": 0,
            "lr": 1,
            "num_epochs": 40,  # Changed from 55 in original
            # "early_stopping_pat": 10,
            "grad_clip": 10,
            "init_weight": 0.04,  # Hacky way to include this for replication, this prob. won't be used anywhere else
            "optimizer_class": optim.SGD,
            "scheduler_class": scheduler.MultiStepLR,
            "scheduler_step_or_epoch": "epoch",
            "scheduler_kwargs": {
                "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
                "milestones": torch.LongTensor(range(13, 54, 1)),
            },
        },
        "lstm_ensemble": {
            "early_stopping": True,
            "weight_decay": 0,
            "lr": 1,
            "num_epochs": 40,  # Changed from 55 in original
            # "early_stopping_pat": 10,
            "grad_clip": 10,
            "init_weight": 0.04,  # Hacky way to include this for replication, this prob. won't be used anywhere else
            "optimizer_class": optim.SGD,
            "scheduler_class": scheduler.MultiStepLR,
            "scheduler_step_or_epoch": "epoch",
            "scheduler_kwargs": {
                "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
                "milestones": torch.LongTensor(range(13, 54, 1)),
            },
        },
        "bayesian_lstm": {
            "early_stopping": True,
            "weight_decay": 0,
            "lr": 1,
            "num_epochs": 40,  # Changed from 55 in original
            # "early_stopping_pat": 10,
            "grad_clip": 10,
            "optimizer_class": optim.SGD,
            "scheduler_class": scheduler.MultiStepLR,
            "scheduler_step_or_epoch": "epoch",
            "scheduler_kwargs": {
                "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
                "milestones": torch.LongTensor(range(13, 54, 1)),
            },
        },
        "st_tau_lstm": {
            "early_stopping": True,
            "weight_decay": 0,
            "lr": 1,
            "num_epochs": 40,  # Changed from 55 in original
            # "early_stopping_pat": 10,
            "grad_clip": 10,
            "optimizer_class": optim.SGD,
            "scheduler_class": scheduler.MultiStepLR,
            "scheduler_step_or_epoch": "epoch",
            "scheduler_kwargs": {
                "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
                "milestones": torch.LongTensor(range(13, 54, 1)),
            },
        },
        # Taken from  https://github.com/yaringal/BayesianRNN/blob/master/LM_code/main_new_dropout_SOTA.lua
        "variational_lstm": {
            "early_stopping": True,
            "weight_decay": 1e-7,
            "lr": 1,
            "num_epochs": 55,  # Changed from 55 in original
            # "early_stopping_pat": 10,
            "grad_clip": 10,
            "init_weight": 0.04,  # Hacky way to include this for replication, this prob. won't be used anywhere else
            "scheduler_class": scheduler.MultiStepLR,
            "scheduler_step_or_epoch": "epoch",
            "scheduler_kwargs": {
                "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
                "milestones": torch.LongTensor(range(13, 54, 1)),
            },
        },
        "composer": {
            "lr": 0.05,
            "num_epochs": 55,
            "grad_clip": 10,
            "optimizer_class": optim.Adam,
            "scheduler_class": transformers.get_linear_schedule_with_warmup,
            "scheduler_step_or_epoch": "step",
            "scheduler_kwargs": {
                # Warmup prob: 0.1, training steps: 469
                "num_warmup_steps": 469 * 80 * 0.1,
                "num_training_steps": 469 * 80,
            },
        },
        "transformer": {
            "lr": 0.05,
            "num_epochs": 55,
            "grad_clip": 10,
            "optimizer_class": optim.Adam,
            "scheduler_class": transformers.get_linear_schedule_with_warmup,
            "scheduler_step_or_epoch": "step",
            "scheduler_kwargs": {
                # Warmup prob: 0.1, training steps: 469
                "num_warmup_steps": 469 * 80 * 0.1,
                "num_training_steps": 469 * 80,
            },
        },
        "due_transformer": {
            "lr": 1e-8,
            "num_epochs": 80,
            "optimizer_class": optim.Adam,
            "scheduler_class": transformers.get_linear_schedule_with_warmup,
            "scheduler_step_or_epoch": "step",
            "scheduler_kwargs": {
                # Warmup prob: 0.1, training steps: 469
                "num_warmup_steps": 469 * 80 * 0.1,
                "num_training_steps": 469 * 80,
            },
        },
    },
    "clinc": {
        "sngp_transformer": {
            "lr": 5e-3,
            "length_scale": 2,
            "weight_decay": 0.1,
            "num_epochs": 40,
            "scheduler_class": transformers.get_linear_schedule_with_warmup,
            "scheduler_step_or_epoch": "step",
            "scheduler_kwargs": {
                # Warmup prob: 0.1, training steps: 469
                "num_warmup_steps": 469 * 40 * 0.1,
                "num_training_steps": 469 * 40,
            },
        },
        "due_transformer": {
            "lr": 5e-3,
            "num_epochs": 80,
            "optimizer_class": optim.Adam,
            "scheduler_class": transformers.get_linear_schedule_with_warmup,
            "scheduler_step_or_epoch": "step",
            "scheduler_kwargs": {
                # Warmup prob: 0.1, training steps: 469
                "num_warmup_steps": 469 * 80 * 0.1,
                "num_training_steps": 469 * 80,
            },
        },
        "ddu_transformer": {
            "lr": 5e-5,
            "num_epochs": 40,
        },
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
            "num_layers": 3,
            "input_dropout": 0.2,
            "dropout": 0.2,
            "num_heads": 5,
            "sequence_length": 30,
            "hidden_size": 768,
            "last_layer_size": 512,
            "spectral_norm_upper_bound": 0.95,
            "ridge_factor": 0.001,
            "scaling_coefficient": 0.999,
            "beta_length_scale": 2,
            "gp_mean_field_factor": 0.1,
            "num_predictions": 10,
        },
    },
    "ptb": {
        "lstm": {
            "num_layers": 2,
            "hidden_size": 650,
            "input_size": 650,
            "dropout": 0.5,
            "vocab_size": 10001,
            "output_size": 10001,
            "is_sequence_classifier": False,
        },
        "bayesian_lstm": {
            "num_layers": 2,
            "hidden_size": 650,
            "input_size": 650,
            "dropout": 0.5,
            "vocab_size": 10001,
            "output_size": 10001,
            "prior_sigma_1": 0.1,
            "prior_sigma_2": 0.002,
            "prior_pi": 1,
            "posterior_mu_init": 0,
            "posterior_rho_init": -6.0,
            "num_predictions": 10,
            "is_sequence_classifier": False,
        },
        "st_tau_lstm": {
            "num_layers": 2,
            "hidden_size": 650,
            "input_size": 650,
            "dropout": 0.5,
            "vocab_size": 10001,
            "output_size": 10001,
            "num_predictions": 10,
            "num_centroids": 20,
            "is_sequence_classifier": False,
        },
        "lstm_ensemble": {
            "num_layers": 2,
            "hidden_size": 650,
            "input_size": 650,
            "dropout": 0.5,
            "vocab_size": 10001,
            "output_size": 10001,
            "ensemble_size": 10,
            "is_sequence_classifier": False,
        },
        # Taken from https://github.com/yaringal/BayesianRNN/blob/master/LM_code/main_new_dropout_SOTA.lua
        "variational_lstm": {
            "num_layers": 2,
            "hidden_size": 1500,
            "input_size": 1500,
            "embedding_dropout": 0.3,  # dropout_x, Large model Gal & Ghrahramani (2016)
            "layer_dropout": 0.5,  # dropout_i / dropout_o, Large model Gal & Ghrahramani (2016)
            "time_dropout": 0.3,  # dropout_h, Large model Gal & Ghrahramani (2016)
            "vocab_size": 10001,
            "output_size": 10001,
            "num_predictions": 250,  # Changed from 1000 because that's just excessive
        },
        "composer": {
            "num_layers": 4,
            "hidden_size": 500,
            "input_size": 500,
            "dropout": 0.2,
            "vocab_size": 10001,
            "output_size": 10001,
            "num_operations": 10,
            "sequence_length": 35,
            "is_sequence_classifier": False,
        },
        "transformer": {
            "num_layers": 6,
            "hidden_size": 500,
            "input_size": 500,
            "vocab_size": 10001,
            "output_size": 10001,
            "input_dropout": 0.2,
            "dropout": 0.2,
            "num_heads": 10,
            "sequence_length": 30,
            "is_sequence_classifier": False,
        },
    },
    "clinc": {
        "sngp_transformer": {
            "num_layers": 3,
            "vocab_size": 10001,
            "output_size": 151,
            "input_size": 500,
            "input_dropout": 0.2,
            "dropout": 0.2,
            "num_heads": 5,
            "sequence_length": 35,
            "hidden_size": 768,
            "last_layer_size": 512,
            "spectral_norm_upper_bound": 0.95,
            "ridge_factor": 0.001,
            "scaling_coefficient": 0.999,
            "beta_length_scale": 2,
            "gp_mean_field_factor": 0.1,
            "num_predictions": 10,
            "is_sequence_classifier": True,
        },
        "due_transformer": {
            "num_layers": 3,
            "hidden_size": 768,
            "num_heads": 5,
            "vocab_size": 10001,
            "output_size": 151,
            "input_dropout": 0.2,
            "dropout": 0.2,
            "num_predictions": 10,
            "num_inducing_points": 20,
            "num_inducing_samples": 10000,
            "spectral_norm_upper_bound": 0.95,
            "kernel_type": "Matern32",
            "input_size": 500,
            "sequence_length": 35,
            "is_sequence_classifier": True,
        },
        "ddu_transformer": {
            "num_layers": 6,
            "hidden_size": 768,
            "num_heads": 10,
            "vocab_size": 10001,
            "output_size": 151,
            "input_dropout": 0.3,
            "dropout": 0.3,
            "input_size": 500,
            "sequence_length": 35,
            "is_sequence_classifier": True,
            "spectral_norm_upper_bound": 0.95,
        },
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
