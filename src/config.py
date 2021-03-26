"""
This module puts all the hyper-, training and preprocessing parameters used in this project into one single place.
"""

# PREPROCESSING PARAMETERS
# List of preprocessing parameters by dataset
SHARED_PREPROCESSING_PARAMS = {"indexing_params": {"min_freq": 20}}
PREPROCESSING_PARAMS = {
    "wikitext-103": {
        "batch_size": 64,
        "sequence_length": 30,
        **SHARED_PREPROCESSING_PARAMS,
    }
}

# TRAINING PARAMETERS
# Training parameters by dataset and model
SHARED_TRAIN_PARAMS = {"wikitext-103": {"num_epochs": 10}}
TRAIN_PARAMS = {
    "wikitext-103": {
        "lstm": {"lr": 0.01, **SHARED_TRAIN_PARAMS["wikitext-103"]},
        "variational_lstm": {"lr": 0.01, **SHARED_TRAIN_PARAMS["wikitext-103"]},
    }
}

# MODEL HYPERPARAMETERS
# Hyperparameters by dataset and model
SHARED_MODEL_PARAMS = {
    "wikitext-103": {
        "input_size": 100,
        "hidden_size": 100,
        "output_size": 20245,
        "vocab_size": 20245,
    }
}
MODEL_PARAMS = {
    "wikitext-103": {
        "lstm": {
            "num_layers": 2,
            "dropout": 0.2,
            **SHARED_MODEL_PARAMS["wikitext-103"],
        },
        "variational_lstm": {
            "num_layers": 2,
            "dropout": 0.35,
            **SHARED_MODEL_PARAMS["wikitext-103"],
        },
    }
}
