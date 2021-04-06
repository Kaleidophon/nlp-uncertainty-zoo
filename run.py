"""
Execute experiments.
"""

# STD
import argparse

# EXT
from torch.utils.tensorboard import SummaryWriter

# PROJECT
from src.config import (
    PREPROCESSING_PARAMS,
    TRAIN_PARAMS,
    MODEL_PARAMS,
    AVAILABLE_DATASETS,
    AVAILABLE_MODELS,
)

# CONST
RESULT_DIR = "./results"
MODEL_DIR = "./models"
DATA_DIR = "./data/processed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=AVAILABLE_DATASETS.keys(),
        help="Dataset to run experiments on.",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        nargs="+",
        choices=AVAILABLE_MODELS.keys(),
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    # Read data
    data = AVAILABLE_DATASETS[args.data](
        data_dir=args.data_dir, **PREPROCESSING_PARAMS[args.data]
    )
    summary_writer = SummaryWriter()

    # TODO: Initialize emissions tracker here
    # TODO: Add number of runs
    # TODO: Add setting of random seed
    # TODO: Wrap in function, add knockknock bot and compile info

    for model_name in args.models:

        model_params = MODEL_PARAMS[args.data][model_name]
        train_params = TRAIN_PARAMS[args.data][model_name]

        module = AVAILABLE_MODELS[model_name](
            model_params, train_params, model_dir="models"
        )
        module.fit(
            train_data=data.train.to(module.device),
            valid_data=data.valid.to(module.device),
        )
