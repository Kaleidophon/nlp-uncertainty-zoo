"""
This module puts all the hyper-, training and preprocessing parameters used in this project into one single place.
"""

# PREPROCESSING PARAMETERS
# List of preprocessing parameters by dataset
SHARED_PREPROCESSING_PARAMS = {"indexing_params": {"min_freq": 20}}
_PREPROCESSING_PARAMS = {"wikitext-103": {"batch_size": 64, "sequence_length": 30}}

# Update shared preprocessing params wth dataset-specific params
PREPROCESSING_PARAMS = {
    dataset: {**SHARED_PREPROCESSING_PARAMS, **preprocessing_params}
    for dataset, preprocessing_params in _PREPROCESSING_PARAMS.items()
}

# TRAINING PARAMETERS
# Training parameters by dataset and model
SHARED_TRAIN_PARAMS = {"wikitext-103": {"num_epochs": 10, "step_size": 1, "gamma": 1}}
_TRAIN_PARAMS = {
    "wikitext-103": {
        "lstm": {"lr": 0.01},
        "variational_lstm": {"lr": 0.01},
        "transformer": {"lr": 0.01, "gamma": 0.95},
    }
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
    }
}
_MODEL_PARAMS = {
    "wikitext-103": {
        "lstm": {"num_layers": 2, "dropout": 0.2},
        "variational_lstm": {"num_layers": 2, "dropout": 0.35},
        "transformer": {
            "num_layers": 6,
            "dropout": 0.2,
            "num_heads": 5,
            "sequence_length": 30,
        },
    }
}
MODEL_PARAMS = {
    dataset: {
        model_name: {**SHARED_MODEL_PARAMS[dataset], **model_params}
        for model_name, model_params in model_dicts.items()
    }
    for dataset, model_dicts in _MODEL_PARAMS.items()
}
