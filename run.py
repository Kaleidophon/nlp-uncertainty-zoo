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
from typing import List, Dict, Optional

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
from src.datasets import LanguageModelingDataset, TextDataset
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
    LanguageModelingDataset: lambda preds, labels: cross_entropy(
        preds, labels, reduction="none"
    )
}
EVAL_FUNCS_POST = {
    LanguageModelingDataset: lambda raw_score: torch.exp(raw_score).item()
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
            score = evaluate(
                model, dataset, f"{result_dir}/{model_name}_{run+1}_{timestamp}.csv"
            )
            scores[model_name].append(score)

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


def evaluate(
    model: Model, dataset: TextDataset, predictions_path: Optional[str] = None
) -> float:
    """
    Evaluate a model and save predictions (if applicable).

    Parameters
    ----------
    model: Model
        Model to be evaluated.
    dataset: TextDataset
        Dataset the model is being evaluated on.
    predictions_path: Optional[str]
        File that predictions are being written to if specified.

    Returns
    -------
    float
        Return score on test set.
    """
    dataset_type = type(dataset).__bases__[0]
    eval_func = EVAL_FUNCS[dataset_type]
    eval_post_func = EVAL_FUNCS_POST[dataset_type]
    prediction_file = None

    if predictions_path is not None:
        prediction_file = codecs.open(predictions_path, "wb", "utf-8")

    cum_scores = 0
    for (X, y) in dataset.test:
        num_seqs = X.shape[0]
        X.to(model.device), y.to(model.device)
        predictions = model.predict(X)

        scores = eval_func(
            rearrange(predictions, "s t p -> (s t) p"),
            rearrange(y, "s l -> (s l)"),
        )
        scores = rearrange(scores, "(s l) -> s l", s=num_seqs)

        if predictions_path is not None:
            for s in range(num_seqs):
                seq, score = dataset.t2i.unindex(X[s, :]), eval_post_func(
                    scores[s, :].mean()
                )
                prediction_file.write(f"{seq}\t{score:.4f}\n")

        cum_scores += scores.mean()

    score = eval_post_func(cum_scores)

    return score


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

    try:
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

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e
