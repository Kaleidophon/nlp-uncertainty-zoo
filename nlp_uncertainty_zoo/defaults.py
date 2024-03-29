"""
Store some default model parameters - for unit testing and quickstart purposes.
"""

# EXT
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import transformers


SEQUENCE_CLASSIFICATION_DEFAULT_PARAMS = {
    "lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 1,
        "num_training_steps": 18480,  # Changed from 55 in original
        "validation_interval": 469,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.04,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)) * 426,
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.5,
        "vocab_size": 28996,
        "output_size": 151,
        "is_sequence_classifier": False,
    },
    "lstm_ensemble": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 1,
        "num_training_steps": 18480,  # Changed from 55 in original
        "validation_interval": 469,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.04,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)) * 426,
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.5,
        "vocab_size": 28996,
        "output_size": 151,
        "ensemble_size": 10,
        "is_sequence_classifier": False,
    },
    "bayesian_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 1,
        "num_training_steps": 18480,  # Changed from 55 in original
        "validation_interval": 469,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)) * 426,
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.5,
        "vocab_size": 28996,
        "output_size": 151,
        "prior_sigma_1": 0.1,
        "prior_sigma_2": 0.002,
        "prior_pi": 1,
        "posterior_mu_init": 0,
        "posterior_rho_init": -6.0,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "st_tau_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 1,
        "num_training_steps": 18480,  # Changed from 55 in original
        "validation_interval": 469,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)) * 426,
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.5,
        "vocab_size": 28996,
        "output_size": 151,
        "num_predictions": 10,
        "num_centroids": 20,
        "is_sequence_classifier": False,
    },
    # Taken from  https://github.com/yaringal/BayesianRNN/blob/master/LM_code/main_new_dropout_SOTA.lua
    "variational_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 1e-7,
        "lr": 1,
        "num_training_steps": 105200,  # Changed from 55 in original
        "validation_interval": 469,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.04,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)) * 426,
        },
        "num_layers": 2,
        "hidden_size": 1500,
        "input_size": 1500,
        "embedding_dropout": 0.3,  # dropout_x, Large model Gal & Ghrahramani (2016)
        "layer_dropout": 0.5,  # dropout_i / dropout_o, Large model Gal & Ghrahramani (2016)
        "time_dropout": 0.3,  # dropout_h, Large model Gal & Ghrahramani (2016)
        "vocab_size": 28996,
        "output_size": 151,
        "num_predictions": 10,  # Changed from 1000 because that's just excessive
    },
    "transformer": {
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 469,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 80 * 0.1,
            "num_training_steps": 469 * 80,
        },
        "num_layers": 6,
        "hidden_size": 500,
        "input_size": 500,
        "vocab_size": 28996,
        "output_size": 151,
        "input_dropout": 0.2,
        "dropout": 0.2,
        "num_heads": 10,
        "sequence_length": 35,
        "is_sequence_classifier": False,
    },
    "variational_transformer": {
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 469,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 80 * 0.1,
            "num_training_steps": 469 * 80,
        },
        "num_layers": 6,
        "hidden_size": 500,
        "input_size": 500,
        "vocab_size": 28996,
        "output_size": 151,
        "input_dropout": 0.2,
        "dropout": 0.2,
        "num_heads": 10,
        "sequence_length": 35,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "variational_bert": {
        "bert_name": "bert-base-uncased",
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 469,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 80 * 0.1,
            "num_training_steps": 469 * 80,
        },
        "output_size": 151,
        "num_predictions": 10,
        "is_sequence_classifier": False,
        "sequence_length": 64,
        "dropout": 0.2,
    },
    "dpp_transformer": {
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 469,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 80 * 0.1,
            "num_training_steps": 469 * 80,
        },
        "num_layers": 6,
        "hidden_size": 500,
        "input_size": 500,
        "vocab_size": 28996,
        "output_size": 151,
        "input_dropout": 0.2,
        "dropout": 0.2,
        "num_heads": 10,
        "sequence_length": 35,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "dpp_bert": {
        "bert_name": "bert-base-uncased",
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 469,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 80 * 0.1,
            "num_training_steps": 469 * 80,
        },
        "output_size": 151,
        "num_predictions": 10,
        "is_sequence_classifier": False,
        "sequence_length": 64,
        "dropout": 0.2,
    },
    "sngp_transformer": {
        "batch_size": 32,
        "lr": 5e-3,
        "length_scale": 2,
        "weight_decay": 0,
        "weight_decay_beta": 0.006236,
        "num_training_steps": 18480,
        "validation_interval": 469,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 40 * 0.1,
            "num_training_steps": 469 * 40,
        },
        "num_layers": 3,
        "vocab_size": 28996,
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
        "num_predictions": 10,
        "kernel_amplitude": 0.01851,
        "is_sequence_classifier": True,
    },
    "sngp_bert": {
        "bert_name": "bert-base-uncased",
        "batch_size": 32,
        "lr": 5e-3,
        "length_scale": 2,
        "weight_decay": 0,
        "weight_decay_beta": 0.006236,
        "num_training_steps": 18480,
        "validation_interval": 469,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 40 * 0.1,
            "num_training_steps": 469 * 40,
        },
        "output_size": 151,
        "last_layer_size": 512,
        "spectral_norm_upper_bound": 0.95,
        "ridge_factor": 0.001,
        "scaling_coefficient": 0.999,
        "beta_length_scale": 2,
        "num_predictions": 10,
        "kernel_amplitude": 0.01851,
        "is_sequence_classifier": True,
    },
    "ddu_transformer": {
        "batch_size": 32,
        "lr": 5e-5,
        "ignore_indices": [-100, 0, 101, 102, 103],
        "num_training_steps": 18480,
        "validation_interval": 469,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 40 * 0.1,
            "num_training_steps": 469 * 40,
        },
        "num_layers": 6,
        "hidden_size": 768,
        "num_heads": 10,
        "vocab_size": 28996,
        "output_size": 151,
        "input_dropout": 0.3,
        "dropout": 0.3,
        "input_size": 500,
        "sequence_length": 35,
        "is_sequence_classifier": True,
        "spectral_norm_upper_bound": 0.95,
        "projection_size": 4,
    },
    "ddu_bert": {
        "bert_name": "bert-base-uncased",
        "ignore_indices": [-100, 0, 101, 102, 103],
        "num_training_steps": 18480,
        "validation_interval": 469,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 40 * 0.1,
            "num_training_steps": 469 * 40,
        },
        "batch_size": 32,
        "lr": 5e-5,
        "output_size": 151,
        "sequence_length": 35,
        "is_sequence_classifier": True,
        "spectral_norm_upper_bound": 0.95,
        "projection_size": 4,
    },
}


LANGUAGE_MODELLING_DEFAULT_PARAMS = {
    "lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 1,
        "num_training_steps": 52600,  # Changed from 55 in original
        "validation_interval": 1315,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.04,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)) * 1315,
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.5,
        "vocab_size": 28996,
        "output_size": 28996,
        "is_sequence_classifier": False,
    },
    "lstm_ensemble": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 1,
        "num_training_steps": 52600,  # Changed from 55 in original
        "validation_interval": 1315,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.04,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)) * 1315,
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.5,
        "vocab_size": 28996,
        "output_size": 28996,
        "ensemble_size": 10,
        "is_sequence_classifier": False,
    },
    "bayesian_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 1,
        "num_training_steps": 52600,  # Changed from 55 in original
        "validation_interval": 1315,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)) * 1315,
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.5,
        "vocab_size": 28996,
        "output_size": 28996,
        "prior_sigma_1": 0.1,
        "prior_sigma_2": 0.002,
        "prior_pi": 1,
        "posterior_mu_init": 0,
        "posterior_rho_init": -6.0,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "st_tau_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 1,
        "num_training_steps": 52600,  # Changed from 55 in original
        "validation_interval": 1315,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)) * 1315,
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.5,
        "vocab_size": 28996,
        "output_size": 28996,
        "num_predictions": 10,
        "num_centroids": 20,
        "is_sequence_classifier": False,
    },
    # Taken from  https://github.com/yaringal/BayesianRNN/blob/master/LM_code/main_new_dropout_SOTA.lua
    "variational_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 1e-7,
        "lr": 1,
        "num_training_steps": 105200,  # Changed from 55 in original
        "validation_interval": 1315,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.04,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)) * 1315,
        },
        "num_layers": 2,
        "hidden_size": 1500,
        "input_size": 1500,
        "embedding_dropout": 0.3,  # dropout_x, Large model Gal & Ghrahramani (2016)
        "layer_dropout": 0.5,  # dropout_i / dropout_o, Large model Gal & Ghrahramani (2016)
        "time_dropout": 0.3,  # dropout_h, Large model Gal & Ghrahramani (2016)
        "vocab_size": 28996,
        "output_size": 28996,
        "num_predictions": 10,  # Changed from 1000 because that's just excessive
    },
    "transformer": {
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 1315,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 80 * 0.1,
            "num_training_steps": 1315 * 80,
        },
        "num_layers": 6,
        "hidden_size": 500,
        "input_size": 500,
        "vocab_size": 28996,
        "output_size": 28996,
        "input_dropout": 0.2,
        "dropout": 0.2,
        "num_heads": 10,
        "sequence_length": 35,
        "is_sequence_classifier": False,
    },
    "ddu_transformer": {
        "batch_size": 32,
        "lr": 0.05,
        "ignore_indices": [-100, 0, 101, 102, 103],
        "num_training_steps": 105200,
        "validation_interval": 1315,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 469 * 80 * 0.1,
            "num_training_steps": 1315 * 80,
        },
        "num_layers": 6,
        "hidden_size": 768,
        "num_heads": 10,
        "vocab_size": 28996,
        "output_size": 28996,
        "input_dropout": 0.3,
        "dropout": 0.3,
        "input_size": 500,
        "sequence_length": 35,
        "is_sequence_classifier": True,
        "spectral_norm_upper_bound": 0.95,
        "projection_size": 2,
    },
    "ddu_bert": {
        "bert_name": "bert-base-uncased",
        "ignore_indices": [-100, 0, 101, 102, 103],
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 1315,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 1315 * 80 * 0.1,
            "num_training_steps": 1315 * 80,
        },
        "output_size": 28996,
        "is_sequence_classifier": True,
        "spectral_norm_upper_bound": 0.95,
        "projection_size": 4,
    },
    "variational_transformer": {
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 1315,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 1315 * 80 * 0.1,
            "num_training_steps": 1315 * 80,
        },
        "num_layers": 6,
        "hidden_size": 500,
        "input_size": 500,
        "vocab_size": 28996,
        "output_size": 28996,
        "input_dropout": 0.2,
        "dropout": 0.2,
        "num_heads": 10,
        "sequence_length": 35,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "variational_bert": {
        "bert_name": "bert-base-uncased",
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 1315,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 1315 * 80 * 0.1,
            "num_training_steps": 1315 * 80,
        },
        "output_size": 28996,
        "num_predictions": 10,
        "is_sequence_classifier": False,
        "dropout": 0.2
    },
    "sngp_transformer": {
        "batch_size": 32,
        "lr": 5e-3,
        "length_scale": 2,
        "weight_decay": 0,
        "weight_decay_beta": 0.006236,
        "num_training_steps": 52600,
        "validation_interval": 1315,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 1315 * 40 * 0.1,
            "num_training_steps": 1315 * 40,
        },
        "num_layers": 3,
        "vocab_size": 28996,
        "output_size": 28996,
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
        "kernel_amplitude": 0.01851,
        "num_predictions": 10,
        "is_sequence_classifier": True,
    },
    "sngp_bert": {
        "bert_name": "bert-base-uncased",
        "batch_size": 32,
        "lr": 5e-3,
        "length_scale": 2,
        "weight_decay": 0,
        "weight_decay_beta": 0.006236,
        "num_training_steps": 52600,
        "validation_interval": 1315,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 1315 * 40 * 0.1,
            "num_training_steps": 1315 * 40,
        },
        "output_size": 28996,
        "last_layer_size": 768,
        "spectral_norm_upper_bound": 0.95,
        "ridge_factor": 0.001,
        "scaling_coefficient": 0.999,
        "beta_length_scale": 2,
        "kernel_amplitude": 0.01851,
        "num_predictions": 10,
        "is_sequence_classifier": True,
    },
    "dpp_transformer": {
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 1315,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 1315 * 80 * 0.1,
            "num_training_steps": 1315 * 80,
        },
        "num_layers": 6,
        "hidden_size": 500,
        "input_size": 500,
        "vocab_size": 28996,
        "output_size": 28996,
        "input_dropout": 0.2,
        "dropout": 0.2,
        "num_heads": 10,
        "sequence_length": 35,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "dpp_bert": {
        "bert_name": "bert-base-uncased",
        "batch_size": 32,
        "lr": 0.05,
        "num_training_steps": 105200,
        "validation_interval": 1315,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1, training steps: 469
            "num_warmup_steps": 1315 * 80 * 0.1,
            "num_training_steps": 1315 * 80,
        },
        "output_size": 28996,
        "num_predictions": 10,
        "is_sequence_classifier": False,
        "dropout": 0.2
    },
}