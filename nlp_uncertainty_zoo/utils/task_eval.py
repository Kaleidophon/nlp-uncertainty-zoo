"""
Implementation of evaluation logic.
"""

# STD
import codecs

# EXT
from einops import rearrange
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import nll_loss
from typing import Optional
from transformers import PreTrainedTokenizerBase

# Map from dataset class to evaluation function
EVAL_FUNCS = {
    "language_modelling": lambda preds, labels: nll_loss(
        torch.log(preds), labels, reduction="none"
    ),
    "sequence_classification": lambda preds, labels: (
        torch.argmax(preds, dim=-1) == labels
    ).long(),
    "token_classification": lambda preds, labels: (
        torch.argmax(preds, dim=-1) == labels
    ).long(),
}
EVAL_FUNCS_POST = {
    "language_modelling": lambda raw_score: torch.exp(raw_score).item(),
    "sequence_classification": lambda raw_score: raw_score.item(),
    "token_classification": lambda raw_score: raw_score.item(),
}


def evaluate(
    model,
    eval_split: DataLoader,
    task: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    predictions_path: Optional[str] = None,
) -> float:
    """
    Evaluate a model and save predictions (if applicable).

    Parameters
    ----------
    model: Model
        Model to be evaluated.
    eval_split: DataSplit
        Data split the model is being evaluated on.
    task: str
        Task type, specified using a string.
    tokenizer: Optional[PreTrainedTokenizerBase]
        Tokenizer of the evaluated model. If given and predictions_path is specified, the input_ids of (sub-word) tokens
        are turned back into strings and saved.
    predictions_path: Optional[str]
        File that predictions are being written to if specified.

    Returns
    -------
    float
        Return score on test set.
    """
    assert (
        task in EVAL_FUNCS
    ), f"Invalid task '{task}' given, must be one of {', '.join(EVAL_FUNCS.keys())}."

    eval_func = EVAL_FUNCS[task]
    eval_post_func = EVAL_FUNCS_POST[task]
    prediction_file = None

    if predictions_path is not None:
        prediction_file = codecs.open(predictions_path, "wb", "utf-8")

    cum_scores = 0
    norm = 0  # Keep track of the number of tokens evaluated
    for batch in eval_split:
        attention_mask, input_ids, labels = (
            batch["attention_mask"].to(model.device),
            batch["input_ids"].to(model.device),
            batch["y"].to(model.device),
        )

        batch_size = labels.shape[0]
        seq_len = 1 if task == "sequence_classification" else input_ids.shape[1]
        predictions = model.predict(input_ids, attention_mask=attention_mask)
        predictions = rearrange(predictions, "b t p -> (b t) p")

        if task in ("language_modelling", "token_classification"):
            labels = rearrange(labels, "b l -> (b l)")

        scores = eval_func(predictions, labels)
        cum_scores += scores.sum()
        norm += batch_size * seq_len

        # Reshape into token scores per sequence
        scores = rearrange(scores, "(b l) -> b l", b=batch_size)

        if predictions_path is not None:
            for s in range(batch_size):
                seq = input_ids[s, :]

                if tokenizer is not None:
                    seq = tokenizer.decode(seq)

                score = scores[s, :]
                prediction_file.write(f"{seq}\t{score:.4f}\n")

    score = eval_post_func(cum_scores / norm)

    if predictions_path is not None:
        prediction_file.write(f"Total score: {score:.4f}\n")
        prediction_file.close()

    return score
