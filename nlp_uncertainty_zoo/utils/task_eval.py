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
from nlp_uncertainty_zoo.datasets import (
    LanguageModelingDataset,
    TextDataset,
    SequenceClassificationDataset,
    DataSplit,
)

# Map from dataset class to evaluation function
EVAL_FUNCS = {
    LanguageModelingDataset: lambda preds, labels: nll_loss(
        torch.log(preds), labels, reduction="none"
    ),
    SequenceClassificationDataset: lambda preds, labels: (
        torch.argmax(preds, dim=-1) == labels
    ).long(),
}
EVAL_FUNCS_POST = {
    LanguageModelingDataset: lambda raw_score: torch.exp(raw_score).item(),
    SequenceClassificationDataset: lambda raw_score: raw_score.item(),
}


# TODO: Refactor for new dataset usage


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
        batch_size = y.shape[0]
        seq_len = 1 if dataset_type == SequenceClassificationDataset else X.shape[1]
        X, y = X.to(model.device), y.to(model.device)
        predictions = model.predict(X)
        predictions = rearrange(predictions, "b t p -> (b t) p")

        if dataset_type == LanguageModelingDataset:
            y = rearrange(y, "b l -> (b l)")

        scores = eval_func(predictions, y)
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
