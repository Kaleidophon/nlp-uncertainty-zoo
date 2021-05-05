"""
Execute experiments.
"""

# STD
import argparse
import codecs
from collections import defaultdict
from datetime import datetime
import json
import os
from typing import List, Dict, Optional, Tuple

# EXT
from codecarbon import OfflineEmissionsTracker
from einops import rearrange
from knockknock import telegram_sender
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter

# PROJECT
from src.model import Model
from src.datasets import DataSplit, LanguageModelingDataset, TextDataset
from src.config import (
    PREPROCESSING_PARAMS,
    TRAIN_PARAMS,
    MODEL_PARAMS,
    AVAILABLE_DATASETS,
    AVAILABLE_MODELS,
)
from src.types import Device

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
    ).item()
}


try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

except ImportError:
    raise ImportError(
        "secret.py wasn't found, please rename secret_template.py and fill in the information."
    )


def run_experiments(
    model_names: List[str],
    dataset: TextDataset,
    runs: int,
    seed: int,
    device: Device,
    model_dir: str,
    result_dir: str,
    summary_writer: Optional[SummaryWriter] = None,
) -> str:
    """
    Run experiments. An experiment consists of training evaluating a number of models on a dataset and savng
    the models and model outputs.

    Parameters
    ----------
    model_names: List[str]
        Names of models that experiments should be run for.
    dataset: TextDataset
        Dataset the model should be run on.
    runs: int
        Number of runs with different random seeds per model.
    seed: int
        Initial seed for every model.
    device: Device
        Device the model is being trained on.
    model_dir: str
        Directory that models are being saved to.
    result_dir: str
        Directory where results will be written to.
    summary_writer: Optional[SummaryWriter]
        Summary writer to track training statistics. Training and validation loss (if applicable) are tracked by
        default, everything else is defined in _epoch_iter() and _finetune() depending on the model.

    Returns
    -------
    Dict[str, Any]
        Information about experiments that is being sent by knockknock.
    """
    scores = defaultdict(list)

    for model_name in model_names:

        np.random.seed(seed)
        torch.manual_seed(seed)

        for run in range(runs):
            timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))

            model_params = MODEL_PARAMS[dataset.name][model_name]
            train_params = TRAIN_PARAMS[dataset.name][model_name]

            model = AVAILABLE_MODELS[model_name](
                model_params, train_params, model_dir=model_dir, device=device
            )
            model.fit(
                train_data=dataset.train,
                valid_data=dataset.valid,
                summary_writer=summary_writer,
            )

            # Evaluate
            model.module.eval()
            seqs, preds, labels = get_predictions(model, dataset.test)
            total_loss = save_predictions(
                f"{result_dir}/{model_name}_{run+1}_{timestamp}.csv",
                dataset,
                seqs,
                preds,
                labels,
            )
            scores[model_name].append(total_loss)

    return json.dumps(
        {
            "dataset": dataset.name,
            "runs": runs,
            "scores": {
                model_name: f"{np.mean(model_scores):.2f} ±{np.std(model_scores):.2f}"
                for model_name, model_scores in scores.items()
            },
        },
        indent=4,
        ensure_ascii=False,
    )


def get_predictions(
    model: Model, test_split: DataSplit
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    batch_seqs, batch_preds, batch_labels = [], [], []

    for (X, y) in test_split:
        X, y = X.to(model.device), y.to(model.device)
        batch_seqs.append(X)
        preds = model.predict(X)
        batch_preds.append(preds)
        batch_labels.append(y)

    batch_seqs = torch.cat(batch_seqs, dim=0)
    batch_preds = torch.cat(batch_preds, dim=0)
    batch_labels = torch.cat(batch_labels, dim=0)

    return batch_seqs, batch_preds, batch_labels


def save_predictions(
    prediction_path: str,
    dataset: TextDataset,
    sequences: torch.LongTensor,
    predictions: torch.FloatTensor,
    labels: torch.LongTensor,
) -> float:
    """
    Save predictions to a file.

    Parameters
    ----------
    prediction_path: str
        Path to which predictions should be saved.
    dataset: TextDataset
        Original dataset.
    sequences: torch.LongTensor
        Sequences in test set.
    predictions: torch.FloatTensor
        Predictions for sequences in test set.
    labels: torch.LongTensor
        Labels for sequences in test set.

    Returns
    -------
    float
        Score on test set.
    """
    eval_func = EVAL_FUNCS[type(dataset).__bases__[0]]

    with codecs.open(prediction_path, "wb", "utf-8") as prediction_file:

        scores = eval_func(
            rearrange(predictions, "s t p -> (s t) p"),
            rearrange(labels, "s l -> (s l)"),
        )
        prediction_file.write(f"Total loss: {scores:.4f}\n")

        for i in range(sequences.shape[0]):
            seq, preds, lbls = sequences[i, :], predictions[i, :, :], labels[i, :]
            original_seq = dataset.t2i.unindex(seq)
            seq_loss = eval_func(preds, lbls)
            prediction_file.write(f"{original_seq}\t{seq_loss:.4f}\n")

    return scores


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
    parser.add_argument("--runs", type=int, default=4)
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

    run_experiments(
        args.models,
        data,
        args.runs,
        args.seed,
        args.device,
        args.model_dir,
        args.result_dir,
        summary_writer,
    )

    if tracker is not None:
        tracker.stop()
