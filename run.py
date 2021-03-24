"""
Execute experiments.
"""

# STD
import argparse

# PROJECT
from src.datasets import Wikitext103Dataset

# CONST
RESULT_DIR = "./results"
MODEL_DIR = "./models"
DATA_DIR = "./data/processed"
AVAILABLE_DATASETS = {"wikitext-103": Wikitext103Dataset}


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
        nargs="+",
        # choices= ... TODO: Add available models here
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    # Read data
    # TODO: Put batch size and sequence length into config
    data = AVAILABLE_DATASETS[args.data](
        data_dir=args.data_dir, batch_size=5, sequence_length=12
    )
