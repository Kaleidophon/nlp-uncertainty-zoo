"""
Define all training and model parameters used for the Wikitext-103 dataset.
"""

# EXT
import numpy as np

WIKITEXT_PREPROCESSING_PARAMS = {
    "batch_size": 64,
    "sequence_length": 30,
    "min_freq": 20,
}

WIKITEXT_TRAIN_PARAMS = {
    "lstm": {"lr": 0.01},
    "variational_lstm": {"lr": 0.01},
    "transformer": {"lr": 0.01, "gamma": 0.95},
    "variational_transformer": {"lr": 0.01, "gamma": 0.95, "weight_decay": 0},
    "sngp_transformer": {"lr": 0.2, "gamma": 0.6, "weight_decay": 0.01},
    "ddu_transformer": {"lr": 0.01, "gamma": 0.95},
}

WIKITEXT_SHARED_MODEL_PARAMS = {
    "input_size": 100,
    "hidden_size": 100,
    "output_size": 20245,
    "vocab_size": 20245,
}

WIKITEXT_MODEL_PARAMS = {
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
}

WIKITEXT_NUM_EVALS = {
    "lstm": 2,
    "variational_lstm": 20,
    "transformer": 2,
    "variational_transformer": 10,
    "sngp_transformer": 10,
    "ddu_transformer": 10,
}

WIKITEXT_PARAM_SEARCH = {
    "lstm": {"num_layers": list(range(2, 6)), "dropout": np.random.uniform(0.1, 0.4)},
    "variational_lstm": {},
    "transformer": {
        "num_layers": list(range(2, 6)),
        "dropout": np.random.uniform(0.1, 0.4),
        "num_heads": [5, 10, 15],
    },
    "variational_transformer": {},
    "sngp_transformer": {},
    "ddu_transformer": {},
}
