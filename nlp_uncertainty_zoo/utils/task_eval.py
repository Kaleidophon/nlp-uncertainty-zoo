"""
Implementation of evaluation logic.
"""

# STD
from collections import defaultdict
from typing import Dict, Tuple

# EXT
import numpy as np
from einops import rearrange
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_task(
    model,
    eval_split: DataLoader,
    ignore_token_ids: Tuple[int] = (-100,),
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a model and save predictions (if applicable).

    Parameters
    ----------
    model: Model
        Model to be evaluated.
    eval_split: DataSplit
        Data split the model is being evaluated on.
    ignore_token_ids: Tuple[int]
        IDs of tokens that should be ignored by the model during evaluation.
    verbose: bool
        Whether to display information about the current progress.

    Returns
    -------
    Dict[str, float]
        Return score on test set.
    """
    scores = defaultdict(float)

    num_batches = len(eval_split)
    progress_bar = tqdm(total=num_batches if verbose else None)

    split_predictions = []
    split_labels = []

    for i, batch in enumerate(eval_split):
        attention_mask, input_ids, labels = (
            batch["attention_mask"].to(model.device),
            batch["input_ids"].to(model.device),
            batch["labels"].to(model.device),
        )

        if len(labels.shape) == 2:
            batch_size, seq_len = labels.shape
        else:
            batch_size, seq_len = labels.shape[0], 1

        with torch.no_grad():
            predictions = model.predict(input_ids, attention_mask=attention_mask)
            predictions = rearrange(predictions, "b t p -> (b t) p")

        if seq_len > 1:
            labels = rearrange(labels, "b l -> (b l)")

        # Filter irrelevant tokens for language modelling / sequence labelling / token predictions
        batch_mask = rearrange(
            torch.all(
                torch.stack([input_ids != idx for idx in ignore_token_ids]), dim=0
            ),
            "b s -> (b s)",
        )

        if seq_len > 1:
            predictions = predictions[batch_mask]
            labels = labels[batch_mask]

            predictions = predictions[labels != -100]
            labels = labels[labels != -100]

        split_predictions.append(np.argmax(predictions.detach().cpu().numpy(), axis=-1))
        split_labels.append(labels.detach().cpu().numpy())

        if verbose:
            progress_bar.set_description(f"Evaluating batch {i+1}/{num_batches}...")
            progress_bar.update(1)

    split_predictions = np.concatenate(split_predictions, axis=0)
    split_labels = np.concatenate(split_labels, axis=0)

    scores["accuracy"] = accuracy_score(split_labels, split_predictions)
    scores["macro_f1_scores"] = f1_score(split_labels, split_predictions, average="macro")

    return scores
