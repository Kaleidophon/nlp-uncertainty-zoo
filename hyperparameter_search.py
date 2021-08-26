"""
Perform hyperparameter search.
"""

# STD
import argparse
from datetime import datetime
from functools import partial
import json
import os
from typing import List, Optional, Dict, Any

# EXT
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.logger import TBXLogger
import torch
from tqdm import tqdm

# PROJECT
from nlp_uncertainty_zoo.config import (
    AVAILABLE_MODELS,
    AVAILABLE_DATASETS,
    PREPROCESSING_PARAMS,
    MODEL_PARAMS,
    TRAIN_PARAMS,
    NUM_EVALS,
    PARAM_SEARCH,
)
from nlp_uncertainty_zoo.utils.evaluation import evaluate
from nlp_uncertainty_zoo.utils.types import Device
from nlp_uncertainty_zoo.datasets import Dataset

# CONST
SEED = 123
HYPERPARAM_DIR = "./hyperparameters"
MODEL_DIR = "./models"
DATA_DIR = "./data/processed"
EMISSION_DIR = "./emissions"
SECRET_IMPORTED = False


try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

    SECRET_IMPORTED = True

except ImportError:
    raise ImportError(
        "secret.py wasn't found, please rename secret_template.py and fill in the information."
    )


def perform_hyperparameter_search(
    models: List[str],
    dataset_name: str,
    max_num_epochs: int,
    result_dir: str,
    device: str = "cpu",
) -> str:
    """
    Perform hyperparameter search for a list of models and save the results into a directory.

    Parameters
    ----------
    models: List[str]
        List specifying the names of models.
    dataset_name: str
        Name of data set models should be evaluated on.
    max_num_epochs: int
        Maximum number of epochs before trial is stopped.
    result_dir: str
        Directory that results should be saved to.
    device: Device
        Device hyperparameter search happens on.

    Returns
    -------
    str
        Information being passed on to knockknock.
    """
    info_dict = {}
    dataset = AVAILABLE_DATASETS[dataset_name](
        data_dir=args.data_dir, **PREPROCESSING_PARAMS[dataset_name]
    )
    # Somehow there's an obscure error if data splits are not loaded in advance, but during hyperparameter search,
    # so do that here
    _, _ = dataset.train, dataset.valid

    def _init_and_train_model(
        config: Dict[str, Any],
        model_name: str,
        train_params: Dict[str, Any],
        device: Device,
        dataset: Optional[Dataset] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize and train a model for hyperparameter search.

        Parameters
        ----------
        config: Dict[str, Any]
            Dictionary of model parameters, including some training parameters that should be optimized.
        model_name: str
            Name of model.
        train_params: Dict[str, Any]
            Dictionary of training parameters.
        device: Device
            Device the model is being trained on.
        dataset: Dataset
            The dataset the model is being trained and evaluated on.
        checkpoint_dir: Optional[str]
            Checkpoint directory. Intentionally set to None so no checkpoints will be made.
        """
        # Because I divide the config in model_params and train_params, but rays.tune only uses a single config,
        # just push train_params that are supposed to be tuned (e.g. the learning rate) into config, then remove
        # them here and overwrite the default param values in train_params
        for param_name, param_value in config.items():
            if param_name in train_params:
                train_params[param_name] = param_value
                del config[param_name]

        model = AVAILABLE_MODELS[model_name](config, train_params, device=device)

        for epoch in range(train_params["num_epochs"]):
            model.epoch_iter(epoch, dataset.train)
            val_score = evaluate(model, dataset, dataset.valid)
            tune.report(val_score=val_score)

    with tqdm(total=len(models)) as progress_bar:

        for model_name in models:
            progress_bar.postfix = f"(model: {model_name})"
            progress_bar.update()
            train_params = TRAIN_PARAMS[dataset_name][model_name]
            config = MODEL_PARAMS[dataset_name][model_name]
            config.update(PARAM_SEARCH[dataset_name][model_name])

            scheduler = ASHAScheduler(
                max_t=max_num_epochs,
                grace_period=1,
                reduction_factor=2,
            )
            # bayesopt = BayesOptSearch()

            # Set up gpus
            gpus_config = {}
            if torch.cuda.is_available() and device == "cuda":
                gpus_config["resources_per_trial"] = {"gpu": torch.cuda.device_count()}

            analysis = tune.run(
                # Wrap function using tune.with_parameters to avoid sending errors due to dataset size
                tune.with_parameters(
                    # Use partial to create a function that only has a config and checkpoint_dir argument
                    partial(
                        _init_and_train_model,
                        model_name=model_name,
                        train_params=train_params,
                        device=device,
                    ),
                    dataset=dataset,
                ),
                config=config,
                loggers=[TBXLogger],
                num_samples=NUM_EVALS[dataset_name][model_name],
                reuse_actors=True,
                scheduler=scheduler,
                # search_alg=bayesopt,
                verbose=3,
                metric="val_score",
                mode="min",
                **gpus_config,
            )

            # Add info for knockknock bot
            info_dict[model_name] = {
                "best_params": analysis.best_config,
                "best_score": analysis.best_result,
            }

            # Add save results
            timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
            results_df = analysis.dataframe()
            results_df.to_csv(
                os.path.join(result_dir, dataset_name, f"{model_name}_{timestamp}.csv")
            )

            progress_bar.update(1)

    if tracker is not None:
        tracker.stop()
        info_dict["emissions"] = tracker._prepare_emissions_data().emissions

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
        "--models",
        type=str,
        required=True,
        nargs="+",
        choices=AVAILABLE_MODELS.keys(),
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--hyperparam-dir", type=str, default=HYPERPARAM_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-top-n", type=int, default=10)
    parser.add_argument("--emission-dir", type=str, default=EMISSION_DIR)
    parser.add_argument("--track-emissions", action="store_true", default=False)
    parser.add_argument("--max-num-epochs", type=int)
    parser.add_argument("--knock", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

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
        args.models,
        args.dataset,
        args.max_num_epochs,
        args.hyperparam_dir,
        args.device,
    )
