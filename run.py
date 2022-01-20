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
import wandb

# PROJECT
from nlp_uncertainty_zoo.utils.task_eval import evaluate
from nlp_uncertainty_zoo.config import (
    MODEL_PARAMS,
    AVAILABLE_DATASETS,
    AVAILABLE_MODELS,
    DATASET_TASKS,
)
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun

# CONST
SEED = 123
RESULT_DIR = "./results"
MODEL_DIR = "./models"
DATA_DIR = "./data/processed"
EMISSION_DIR = "./emissions"
PROJECT_NAME = "nlp-uncertainty-zoo"

# GLOBALS
SECRET_IMPORTED = False

# Knockknock support
try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

    SECRET_IMPORTED = True

except ImportError:
    pass

# CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def run_experiments(
    model_names: List[str],
    dataset_name: str,
    runs: int,
    seed: int,
    device: Device,
    data_dir: str,
    model_dir: str,
    result_dir: str,
    wandb_run: Optional[WandBRun] = None,
) -> str:
    """
    Run experiments. An experiment consists of training evaluating a number of models on a dataset and saving
    the models and model outputs.

    Parameters
    ----------
    model_names: List[str]
        Names of models that experiments should be run for.
    dataset_name: str
        Name of the dataset the model should be run on.
    runs: int
        Number of runs with different random seeds per model.
    seed: int
        Initial seed for every model.
    device: Device
        Device the model is being trained on.
    data_dir: str
        Directory the data is stored in.
    model_dir: str
        Directory that models are being saved to.
    result_dir: str
        Directory where results will be written to.
    wandb_run: Optional[WandBRun]
        Weights and Biases Run to track training statistics. Training and validation loss (if applicable) are tracked by
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

        # Get model (hyper-)parameters
        model_params = MODEL_PARAMS[dataset_name][model_name]

        # Read data and build data splits
        dataset_task = DATASET_TASKS[dataset_name]
        dataset_builder = AVAILABLE_DATASETS[dataset_name](
            data_dir=data_dir, max_length=model_params["sequence_length"]
        )
        data_splits = dataset_builder.build(batch_size=model_params["batch_size"])

        for run in range(runs):
            timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))

            model = AVAILABLE_MODELS[model_name](
                model_params, model_dir=model_dir, device=device
            )

            result_dict = model.fit(
                train_split=data_splits["train"],
                valid_split=data_splits["valid"],
                wandb_run=wandb_run,
            )

            # Evaluate
            model.module.eval()
            score = evaluate(
                model,
                eval_split=data_splits["test"],
                task=dataset_task,
                tokenizer=dataset_builder.tokenizer,
                predictions_path=f"{result_dir}/{model_name}_{run+1}_{timestamp}.csv",
            )
            scores[model_name].append(score)

            # Add all info to Weights & Biases
            if wandb_run is not None:
                wandb_run.config = model_params
                wandb_run.log(
                    {
                        "train_loss": result_dict["train_loss"],
                        "best_val_loss": result_dict["best_val_loss"],
                        "test_score": score,
                    }
                )

                # Reset for potential next run
                wandb_run.finish()
                wandb_run = wandb.init(PROJECT_NAME)

    return json.dumps(
        {
            "dataset": dataset_name,
            "runs": runs,
            "scores": {
                model_name: f"{np.mean(model_scores):.2f} ±{np.std(model_scores):.2f}"
                for model_name, model_scores in scores.items()
            },
            "url": wandb.run.get_url(),
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

    wandb_run = wandb.init(project=PROJECT_NAME)
    tracker = None

    if args.track_emissions:
        timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        emissions_path = os.path.join(args.emission_dir, timestamp)
        os.mkdir(emissions_path)
        tracker = OfflineEmissionsTracker(
            project_name="nlp_uncertainty_zoo-experiments",
            country_iso_code=COUNTRY_CODE,
            output_dir=emissions_path,
        )
        tracker.start()

    # Apply decorator
    if args.knock:
        if not SECRET_IMPORTED:
            raise ImportError(
                "secret.py wasn't found, please rename secret_template.py and fill in the information."
            )

        run_experiments = telegram_sender(
            token=TELEGRAM_API_TOKEN, chat_id=TELEGRAM_CHAT_ID
        )(run_experiments)

    try:
        run_experiments(
            args.models,
            args.dataset,
            args.runs,
            args.seed,
            args.device,
            args.data_dir,
            args.model_dir,
            args.result_dir,
            wandb_run,
        )

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e
