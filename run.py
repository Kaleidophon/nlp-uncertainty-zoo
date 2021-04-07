"""
Execute experiments.
"""

# STD
import argparse
from datetime import datetime
import os
from typing import List, Any, Dict, Optional

# EXT
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
import numpy as np
import torch
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
SEED = 123
RESULT_DIR = "./results"
MODEL_DIR = "./models"
DATA_DIR = "./data/processed"
EMISSION_DIR = "./emissions"


try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

except ImportError:
    raise ImportError(
        "secret.py wasn't found, please rename secret_template.py and fill in the information."
    )


@telegram_sender(token=TELEGRAM_API_TOKEN, chat_id=TELEGRAM_CHAT_ID)
def run_experiments(
    model_names: List[str],
    dataset_name: str,
    runs: int,
    seed: int,
    summary_writer: Optional[SummaryWriter] = None,
) -> Dict[str, Any]:
    """
    Run experiments. An experiment consists of training evaluating a number of models on a dataset and savng
    the models and model outputs.

    Parameters
    ----------
    model_names: List[str]
        Names of models that experiments should be run for.
    dataset_name: str
        Name of dataset the model should be run on.
    runs: int
        Number of runs with different random seeds per model.
    seed: int
        Initial seed for every model.
    summary_writer: Optional[SummaryWriter]
        Summary writer to track training statistics. Training and validation loss (if applicable) are tracked by
        default, everything else is defined in _epoch_iter() and _finetune() depending on the model.

    Returns
    -------
    Dict[str, Any]
        Information about experiments that is being sent by knockknock.
    """
    for model_name in model_names:

        np.random.seed(seed)
        torch.manual_seed(seed)

        for run in range(runs):

            model_params = MODEL_PARAMS[dataset_name][model_name]
            train_params = TRAIN_PARAMS[dataset_name][model_name]

            module = AVAILABLE_MODELS[model_name](
                model_params, train_params, model_dir="models"
            )
            module.fit(
                train_data=data.train.to(module.device),
                valid_data=data.valid.to(module.device),
                summary_writer=summary_writer,
            )

            # TODO: Evaluate
            # TODO: Save model predictions
            # TODO: Compile info for knockkock

    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
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
    parser.add_argument("--emission-dir", type=str, default=EMISSION_DIR)
    parser.add_argument("--track-emissions", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--runs", type=str, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # Read data
    data = AVAILABLE_DATASETS[args.data](
        data_dir=args.data_dir, **PREPROCESSING_PARAMS[args.data]
    )

    summary_writer = SummaryWriter()
    tracker = None

    if args.track_emissions:
        timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        emissions_path = os.path.join(args.emission_dir, timestamp)
        os.mkdir(emissions_path)
        tracker = OfflineEmissionsTracker(
            project_name="nlp-uncertainty-zoo-experiments",
            country_iso_code=COUNTRY_CODE,
            output_dir=emissions_path,
        )
        tracker.start()

    run_experiments(
        args.models, args.dataset, args.run, args.seed, summary_writer, tracker
    )

    if tracker is not None:
        tracker.stop()
