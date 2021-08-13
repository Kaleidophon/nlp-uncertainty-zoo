"""
Define all training and model parameters used for the CLINC dataset.
"""

# EXT
import torch.optim as optim
import transformers


CLINC_PREPROCESSING_PARAMS = {"batch_size": 32, "sequence_length": 32}

CLINC_TRAIN_PARAMS = {
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
}

CLINC_SHARED_MODEL_PARAMS = {}

CLINC_MODEL_PARAMS = {
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
}
