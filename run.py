"""
Execute experiments.
"""

# STD
import argparse

# EXT
from torch.utils.tensorboard import SummaryWriter

# PROJECT
from src.config import PREPROCESSING_PARAMS, TRAIN_PARAMS, MODEL_PARAMS
from src.datasets import Wikitext103Dataset
from src.lstm import LSTMModule
from src.dropout import VariationalLSTMModule, VariationalTransformerModule
from src.transformer import TransformerModule

# CONST
RESULT_DIR = "./results"
MODEL_DIR = "./models"
DATA_DIR = "./data/processed"
AVAILABLE_DATASETS = {"wikitext-103": Wikitext103Dataset}
AVAILABLE_MODELS = {
    "lstm": LSTMModule,
    "variational_lstm": VariationalLSTMModule,
    "transformer": TransformerModule,
    "variational_transformer": VariationalTransformerModule,
}


# TODO: Add notification about finished training runs using https://github.com/huggingface/knockknock


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
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    # Read data
    data = AVAILABLE_DATASETS[args.data](
        data_dir=args.data_dir, **PREPROCESSING_PARAMS[args.data]
    )
    summary_writer = SummaryWriter()

    for model_name in args.models:

        model_params = MODEL_PARAMS[args.data][model_name]
        train_params = TRAIN_PARAMS[args.data][model_name]

        module = AVAILABLE_MODELS[model_name](
            model_params, train_params, model_dir="models"
        )
        module.fit(data)
