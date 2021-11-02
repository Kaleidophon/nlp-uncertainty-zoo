"""
Define all training and model parameters used for the Wikitext-103 dataset.
"""

WIKITEXT_PREPROCESSING_PARAMS = {
    "batch_size": 64,
    "sequence_length": 30,
    "min_freq": 20,
}

WIKITEXT_MODEL_PARAMS = {
    "lstm": {"lr": 0.01, "num_layers": 2, "dropout": 0.2},
    "variational_lstm": {"lr": 0.01, "num_layers": 2, "dropout": 0.35},
    "transformer": {
        "lr": 0.01,
        "gamma": 0.95,
        "num_layers": 6,
        "input_dropout": 0.2,
        "dropout": 0.2,
        "num_heads": 5,
        "sequence_length": 30,
        "input_size": 100,
        "hidden_size": 100,
        "output_size": 20245,
        "vocab_size": 20245,
    },
    "variational_transformer": {
        "lr": 0.01,
        "gamma": 0.95,
        "weight_decay": 0,
        "num_layers": 6,
        "input_dropout": 0.2,
        "dropout": 0.2,
        "num_heads": 5,
        "sequence_length": 30,
        "num_predictions": 100,
        "input_size": 100,
        "hidden_size": 100,
        "output_size": 20245,
        "vocab_size": 20245,
    },
    "sngp_transformer": {
        "lr": 0.2,
        "gamma": 0.6,
        "weight_decay": 0.01,
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
        "input_size": 100,
        "output_size": 20245,
        "vocab_size": 20245,
    },
    "ddu_transformer": {"lr": 0.01, "gamma": 0.95},
}
