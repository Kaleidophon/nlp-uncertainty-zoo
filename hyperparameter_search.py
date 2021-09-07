"""
Perform hyperparameter search.
"""

# STD
import argparse
from datetime import datetime
import json
import os
from typing import List, Dict, Union, Optional

# EXT
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
from sklearn.model_selection import ParameterSampler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
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

except ImportError:
    raise ImportError(
        "secret.py wasn't found, please rename secret_template.py and fill in the information."
    )


def perform_hyperparameter_search(
    models: List[str],
    dataset_name: str,
    max_num_epochs: int,
    result_dir: str,
    save_top_n: int = 10,
    device: str = "cpu",
    summary_writer: Optional[SummaryWriter] = None,
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
    save_top_n: int
        Save the top n parameter configuration. Default is 10.
    device: Device
        Device hyperparameter search happens on.
    summary_writer: Optional[SummaryWriter]
        Summary writer to track training statistics. Training and validation loss (if applicable) are tracked by
        default, everything else is defined in _epoch_iter() and _finetune() depending on the model.

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

    with tqdm(total=get_num_runs(models, dataset_name)) as progress_bar:

        for model_name in models:

            progress_bar.postfix = f"(model: {model_name})"
            progress_bar.update()
            scores = {}

            sampled_params = sample_hyperparameters(model_name, dataset_name)

            for run, param_set in enumerate(sampled_params):

                model_params = MODEL_PARAMS[dataset_name][model_name]
                train_params = TRAIN_PARAMS[dataset_name][model_name]
                train_params["num_epochs"] = max_num_epochs

                module = AVAILABLE_MODELS[model_name](
                    model_params, train_params, model_dir="models", device=device
                )

                try:
                    module.fit(
                        dataset.train.to(device),
                        verbose=True,  # TODO
                        summary_writer=summary_writer,
                    )
                    score = -module.eval(dataset.valid.to(device)).item()

                # In case of nans due bad training parameters
                except (ValueError, RuntimeError) as e:
                    print(f"There was an error: '{str(e)}', run aborted.")
                    score = -np.inf

                if np.isnan(score):
                    score = -np.inf

                scores[run] = {"score": score, "hyperparameters": param_set}
                progress_bar.update(1)

                # Rank and save results
                # Do after every experiment in case anything goes wrong
                sorted_scores = dict(
                    list(
                        sorted(
                            scores.items(),
                            key=lambda run: run[1]["score"],
                            reverse=True,
                        )
                    )[:save_top_n]
                )
                model_result_dir = f"{result_dir}/{dataset_name}/"
                info_dict[model_name] = sorted_scores[0]

                if not os.path.exists(model_result_dir):
                    os.makedirs(model_result_dir)

                with open(f"{model_result_dir}/{model_name}.json", "w") as result_file:
                    result_file.write(json.dumps(sorted_scores, indent=4, default=str))

    if tracker is not None:
        tracker.stop()
        info_dict["emissions"] = tracker._prepare_emissions_data().emissions

    return "\n" + json.dumps(info_dict, indent=4)


def get_num_runs(model_names: List[str], dataset_name: str) -> int:
    """
    Calculate the total number of runs for this search given a list of model names.
    """
    return sum([NUM_EVALS[dataset_name][model_name] for model_name in model_names])


def sample_hyperparameters(
    model_name: str, dataset_name: str, round_to: int = 6
) -> List[Dict[str, Union[int, float]]]:
    """
    Sample the hyperparameters for different runs of the same model. The distributions parameters are sampled from are
    defined in nlp_uncertainty_zoo.config.PARAM_SEARCH and the number of evaluations per model type in
    nlp_uncertainty_zoo.config.NUM_EVALS.

    Parameters
    ----------
    model_name: str
        Name of the model.
    dataset_name: str
        Specify the data set which should be used to specify the hyperparameters to be sampled / default values.
    round_to: int
        Decimal that floats should be rounded to.

    Returns
    -------
    sampled_params: List[Dict[str, Union[int, float]]]
        List of dictionaries containing hyperparameters and their sampled values.
    """
    sampled_params = list(
        ParameterSampler(
            param_distributions={
                hyperparam: PARAM_SEARCH[dataset_name][model_name][hyperparam]
                for hyperparam, val in MODEL_PARAMS[dataset_name][model_name].items()
                if hyperparam in PARAM_SEARCH[dataset_name][model_name]
            },
            n_iter=NUM_EVALS[dataset_name][model_name],
        )
    )

    sampled_params = [
        dict(
            {
                # Round float values
                hyperparam: round(val, round_to) if isinstance(val, float) else val
                for hyperparam, val in params.items()
            },
            **{
                # Add hyperparameters that stay fixed
                hyperparam: val
                for hyperparam, val in MODEL_PARAMS[dataset_name][model_name].items()
                if hyperparam not in PARAM_SEARCH[dataset_name][model_name]
            },
        )
        for params in sampled_params
    ]

    return sampled_params


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

    summary_writer = SummaryWriter()
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
        args.save_top_n,
        args.device,
        summary_writer,
    )
