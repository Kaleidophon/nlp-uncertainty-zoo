"""
Execute experiments.
"""

# STD
import argparse
from datetime import datetime
import os
from typing import List, Any, Dict, Optional, Tuple

# EXT
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter

# PROJECT
from src.model import Model
from src.datasets import DataSplit, LanguageModelingDataset
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

# Map from dataset class to evaluation function
EVAL_FUNCS = {
    LanguageModelingDataset: lambda preds, labels: torch.exp(
        cross_entropy(preds, labels)
    )
}


try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

except ImportError:
    raise ImportError(
        "secret.py wasn't found, please rename secret_template.py and fill in the information."
    )


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

            model = AVAILABLE_MODELS[model_name](
                model_params, train_params, model_dir="models"
            )
            model.fit(
                train_data=data.train.to(model.device),
                valid_data=data.valid.to(model.device),
                summary_writer=summary_writer,
            )

            # Evaluate
            model.module.eval()
            preds, labels = get_predictions(model, data.test.to(model.device))
            score = EVAL_FUNCS[type(data).__bases__[0]](preds, labels)
            print(score)
            # TODO: Save model predictions
            # TODO: Compile info for knockkock

            # TODO: Debug
            break

    return {}


def get_predictions(
    model: Model, test_split: DataSplit
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve the predictions from the models for a test split.

    Parameters
    ----------
    model: Model
        Current model.
    test_split: DataSplit
        Test split the model is being evaluated on.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of predictions and targets as tensors.
    """
    batch_preds, batch_labels = [], []

    for (X, y) in test_split:
        preds = model.predict(X.to(model.device))
        batch_preds.append(preds)
        batch_labels.append(y)
        break  # TODO: Debug

    batch_preds = torch.cat(batch_preds, dim=0)
    batch_labels = torch.cat(batch_labels, dim=0)
    num_samples, sequence_length, num_classes = batch_preds.shape
    batch_preds = batch_preds.reshape(num_samples * sequence_length, num_classes)
    batch_labels = batch_labels.reshape(num_samples * sequence_length)

    return batch_preds, batch_labels


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
    parser.add_argument("--knock", action="store_true", default=False)
    args = parser.parse_args()

    # Read data
    data = AVAILABLE_DATASETS[args.dataset](
        data_dir=args.data_dir, **PREPROCESSING_PARAMS[args.dataset]
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

    # Apply decorator
    if args.knock:
        run_experiments = telegram_sender(
            token=TELEGRAM_API_TOKEN, chat_id=TELEGRAM_CHAT_ID
        )(run_experiments)

    run_experiments(args.models, args.dataset, args.runs, args.seed, summary_writer)

    if tracker is not None:
        tracker.stop()
