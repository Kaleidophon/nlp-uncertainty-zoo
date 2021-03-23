"""
Execute experiments.
"""

# STD
import argparse

# CONST
RESULT_DIR = "./results"
MODEL_DIR = "./models"
AVAILABLE_DATASETS = ("wikitext-103",)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Dataset to run experiments on.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        # choices= ... TODO: Add available models here
    )
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)

    args = parser.parse_args()
