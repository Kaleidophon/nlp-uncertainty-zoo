"""
Implementation of evaluation logic.
"""

# STD
import codecs
from collections import defaultdict
from typing import Dict

# EXT
import numpy as np
from einops import rearrange
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import nll_loss
from typing import Optional
from transformers import PreTrainedTokenizerBase


def evaluate(
    model,
    eval_split: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, float]:
    """
    Evaluate a model and save predictions (if applicable).

    Parameters
    ----------
    model: Model
        Model to be evaluated.
    eval_split: DataSplit
        Data split the model is being evaluated on.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer of the evaluated model.

    Returns
    -------
    Dict[str, float]
        Return score on test set.
    """
    scores = defaultdict(float)

    split_predictions = []
    split_labels = []
    for batch in eval_split:
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
        ignore_indices = tokenizer.all_special_ids + [-100]
        batch_mask = rearrange(
            torch.all(
                torch.stack([input_ids != idx for idx in ignore_indices]), dim=0
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

    split_predictions = np.concatenate(split_predictions, axis=0)
    split_labels = np.concatenate(split_labels, axis=0)

    scores["accuracy"] = accuracy_score(split_labels, split_predictions)
    scores["macro_f1_scores"] = f1_score(split_labels, split_predictions, average="macro")

    return scores
