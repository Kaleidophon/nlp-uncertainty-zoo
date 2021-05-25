"""
Execute experiments.
"""

# STD
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from typing import List, Dict, Optional

# EXT
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# PROJECT
from src.datasets import TextDataset
from src.evaluation import evaluate
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
    Run experiments. An experiment consists of training evaluating a number of models on a dataset and saving
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
            # TODO: Debug
            result_dict = {}
            # result_dict = model.fit(
            #    dataset=dataset,
            #    summary_writer=summary_writer,
            # )

            # Evaluate
            model.module.eval()
            score = evaluate(
                model,
                dataset,
                dataset.test,
                f"{result_dir}/{model_name}_{run+1}_{timestamp}.csv",
            )
            scores[model_name].append(score)

            # Add all info to summary writer
            if summary_writer is not None:
                summary_writer.add_hparams(
                    hparam_dict={**model_params, **train_params},
                    metric_dict={
                        "train_loss": result_dict["train_loss"],
                        "best_val_loss": result_dict["best_val_loss"],
                        "test_score": score,
                    },
                )
                # Reset for potential next run
                summary_writer.close()
                summary_writer = SummaryWriter()

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
