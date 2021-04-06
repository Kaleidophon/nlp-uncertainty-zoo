"""
Perform hyperparameter search.
"""

# STD
import argparse
import json
import os
from typing import List, Dict, Union

# EXT
from sklearn.model_selection import ParameterSampler
import numpy as np
import torch
from tqdm import tqdm

# PROJECT
from src.config import (
    AVAILABLE_MODELS,
    AVAILABLE_DATASETS,
    PREPROCESSING_PARAMS,
    MODEL_PARAMS,
    TRAIN_PARAMS,
    NUM_EVALS,
    PARAM_SEARCH,
)

# CONST
SEED = 123
HYPERPARAM_DIR = "./hyperparameters"
MODEL_DIR = "./models"
DATA_DIR = "./data/processed"


def perform_hyperparameter_search(
    dataset_name: str,
    models: List[str],
    result_dir: str,
    save_top_n: int = 10,
    device: str = "cpu",
    track_emissions: bool = True,
):
    """
    Perform hyperparameter search for a list of models and save the results into a directory.

    Parameters
    ----------
    dataset_name: str
        Name of data set models should be evaluated on.
    models: List[str]
        List specifying the names of models.
    result_dir: str
        Directory that results should be saved to.
    save_top_n: int
        Save the top n parameter configuration. Default is 10.
    device: Device
        Device hyperparameter search happens on.
    track_emissions: bool
        Indicate whether carbon emissions should be tracked.
    """

    dataset = AVAILABLE_DATASETS[dataset_name](
        data_dir=args.data_dir, **PREPROCESSING_PARAMS[dataset_name]
    )

    with tqdm(total=get_num_runs(models, dataset_name)) as progress_bar:

        for model_name in models:

            progress_bar.postfix = f"(model: {model_name})"
            progress_bar.update()
            scores = {}

            sampled_params = sample_hyperparameters(model_name, dataset_name)

            for run, param_set in enumerate(sampled_params):

                model_params = MODEL_PARAMS[args.data][model_name]
                train_params = TRAIN_PARAMS[args.data][model_name]

                module = AVAILABLE_MODELS[model_name](
                    model_params, train_params, model_dir="models", device=device
                )

                try:
                    module.fit(
                        dataset.train.to(device),
                        verbose=False,
                        track_emissions=track_emissions,
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

                if not os.path.exists(model_result_dir):
                    os.makedirs(model_result_dir)

                with open(f"{model_result_dir}/{model_name}.json", "w") as result_file:
                    result_file.write(json.dumps(sorted_scores, indent=4, default=str))


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
    defined in src.config.PARAM_SEARCH and the number of evaluations per model type in src.config.NUM_EVALS.

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
    parser.add_argument("--hyperparam-dir", type=str, default=HYPERPARAM_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-top-n", type=int, default=10)
    parser.add_argument("--track-emissions", action="store_true", default=False)

    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    perform_hyperparameter_search(
        args.data,
        args.models,
        args.hyperparam_dir,
        args.save_top_n,
        args.device,
        args.track_emissions,
    )
