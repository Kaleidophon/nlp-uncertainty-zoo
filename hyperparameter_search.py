"""
Perform hyperparameter search.
"""

# STD
import argparse
from datetime import datetime
import json
import os
from typing import Optional, List

# EXT
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
import numpy as np
import torch
import wandb

# PROJECT
from nlp_uncertainty_zoo.config import (
    AVAILABLE_MODELS,
    AVAILABLE_DATASETS,
    PREPROCESSING_PARAMS,
    MODEL_PARAMS,
)
from nlp_uncertainty_zoo.utils.types import Device, WandBRun

# CONST
EMISSION_DIR = "./emissions"
SEED = 123
DATA_DIR = "./data/processed"
SECRET_IMPORTED = False
PROJECT_NAME = "nlp-uncertainty-zoo"


try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

    SECRET_IMPORTED = True

except ImportError:
    raise ImportError(
        "secret.py wasn't found, please rename secret_template.py and fill in the information."
    )


def perform_hyperparameter_search(
    model_name: str,
    dataset_name: str,
    device: Device = "cpu",
    seed: Optional[int] = None,
    wandb_run: Optional[WandBRun] = None,
) -> str:
    """
    Perform hyperparameter search for a list of models and save the results into a directory.

    Parameters
    ----------
    model_name: str
        The name of model to run the search for.
    dataset_name: str
        Name of data set models should be evaluated on.
    device: Device
        Device hyperparameter search happens on.
    seed: Optional[int]
        Seed for the hyperparameter run.
    wandb_run: Optional[WandBRun]
        Weights and Biases Run to track training statistics. Training and validation loss (if applicable) are tracked by
        default, everything else is defined in _epoch_iter() and _finetune() depending on the model.

    Returns
    -------
    str
        Information being passed on to knockknock.
    """
    if seed is not None:
        torch.random.manual_seed(seed)

    info_dict = {}

    if wandb_run is not None:
        info_dict["config"] = wandb_run.config

    dataset = AVAILABLE_DATASETS[dataset_name](
        data_dir=args.data_dir, **PREPROCESSING_PARAMS[dataset_name]
    )
    # Somehow there's an obscure error if data splits are not loaded in advance, but during hyperparameter search,
    # so do that here
    _, _ = dataset.train, dataset.valid

    model_params = MODEL_PARAMS[dataset_name][model_name]

    module = AVAILABLE_MODELS[model_name](model_params, device=device)

    try:
        module.fit(dataset, validate=True, verbose=False, wandb_run=wandb)
        score = -module.eval(dataset.valid.to(device)).item()

    # In case of nans due bad training parameters
    except (ValueError, RuntimeError) as e:
        print(f"There was an error: '{str(e)}', run aborted.")
        score = -np.inf

    if np.isnan(score):
        score = -np.inf

    info_dict["score"] = score
    wandb.log({"score": score})

    if tracker is not None:
        tracker.stop()
        emissions = tracker._prepare_emissions_data().emissions
        info_dict["emissions"] = emissions
        wandb.log({"emissions": emissions})

    info_dict["url"] = wandb.run.get_url()
    return "\n" + json.dumps(info_dict, indent=4)


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
        "--model",
        type=str,
        required=True,
        choices=AVAILABLE_MODELS.keys(),
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--track-emissions", action="store_true", default=False)
    parser.add_argument("--emission-dir", type=str, default=EMISSION_DIR)
    parser.add_argument("--knock", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=SEED)

    # Parse into the arguments specified above, everything else are ran parameters
    args, config = parser.parse_known_args()

    def _stupid_parse(raw_config: List[str]):
        config = {}

        for raw_arg in raw_config:
            arg, value = raw_arg.strip().replace("--", "").split("=")

            try:
                config[arg] = int(value)

            except TypeError:
                try:
                    config[arg] = float(value)

                except TypeError:
                    # Argument is probably a string
                    config[arg] = value

        return config

    config = _stupid_parse(config)
    model_params = MODEL_PARAMS[args.dataset][args.model]
    model_params.update(config)

    wandb_run = wandb.init(project=PROJECT_NAME, config=model_params)
    tracker = None

    if args.track_emissions:
        timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        emissions_path = os.path.join(args.emission_dir, timestamp)
        os.mkdir(emissions_path)
        tracker = OfflineEmissionsTracker(
            project_name="nlp_uncertainty_zoo-hyperparameters",
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

        perform_hyperparameter_search = telegram_sender(
            token=TELEGRAM_API_TOKEN, chat_id=TELEGRAM_CHAT_ID
        )(perform_hyperparameter_search)

    perform_hyperparameter_search(
        args.model, args.dataset, args.device, args.seed, wandb_run
    )
