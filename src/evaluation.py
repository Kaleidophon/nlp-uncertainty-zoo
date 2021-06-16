"""
Implementation of evaluation logic.
"""

# STD
import codecs

# EXT
from einops import rearrange
import torch
from torch.nn.functional import nll_loss
from typing import Optional

# PROJECT
from src.datasets import LanguageModelingDataset, TextDataset, DataSplit


# Map from dataset class to evaluation function
EVAL_FUNCS = {
    LanguageModelingDataset: lambda preds, labels: nll_loss(
        torch.log(preds), labels, reduction="none"
    )
}
EVAL_FUNCS_POST = {
    LanguageModelingDataset: lambda raw_score: torch.exp(raw_score).item()
}


def evaluate(
    model,
    dataset: TextDataset,
    eval_split: DataSplit,
    predictions_path: Optional[str] = None,
) -> float:
    """
    Evaluate a model and save predictions (if applicable).

    Parameters
    ----------
    model: Model
        Model to be evaluated.
    dataset: TextDataset
        Dataset the model the eval split is from.
    eval_split: DataSplit
        Data split the model is being evaluated on.
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
    norm = 0  # Keep track of the number of tokens evaluated
    for (X, y) in eval_split:
        batch_size, seq_len = X.shape
        X, y = X.to(model.device), y.to(model.device)
        predictions = model.predict(X)

        scores = eval_func(
            rearrange(predictions, "b t p -> (b t) p"),
            rearrange(y, "b l -> (b l)"),
        )
        cum_scores += scores.sum()
        norm += batch_size * seq_len

        # Reshape into token scores per sequence
        scores = rearrange(scores, "(b l) -> b l", b=batch_size)

        if predictions_path is not None:
            for s in range(batch_size):
                seq, score = dataset.t2i.unindex(X[s, :]), eval_post_func(
                    scores[s, :].mean()
                )
                prediction_file.write(f"{seq}\t{score:.4f}\n")

    score = eval_post_func(cum_scores / norm)

    if predictions_path is not None:
        prediction_file.write(f"Total score: {score:.4f}\n")
        prediction_file.close()

    return score
