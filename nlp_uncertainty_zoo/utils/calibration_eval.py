"""
Module to implement different metrics the quality of the model's calibration.
"""

# STD
from collections import defaultdict
from typing import Dict, Any, Tuple, Callable

# EXT
from einops import rearrange
from frozendict import frozendict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# PROJECT
from nlp_uncertainty_zoo.models.model import Model


def ece(y_true: np.array, y_pred: np.array, n_bins: int = 10) -> float:
    """

    Calculate the Expected Calibration Error: for each bin, the absolute difference between
    the mean fraction of positives and the average predicted probability is taken. The ECE is
    the weighed mean of these differences.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    n_bins: int
        The number of bins to use.
    Returns
    -------
    ece: float
        The expected calibration error.
    """
    n = len(y_pred)
    bins = np.arange(0.0, 1.0, 1.0 / n_bins)
    y_pred = np.max(y_pred, axis=-1)
    bins_per_prediction = np.digitize(y_pred, bins)

    df = pd.DataFrame({"y_pred": y_pred, "y": y_true, "pred_bins": bins_per_prediction})

    grouped_by_bins = df.groupby("pred_bins")
    # calculate the mean y and predicted probabilities per bin
    binned = grouped_by_bins.mean()

    # calculate the number of items per bin
    binned_counts = grouped_by_bins["y"].count()

    # calculate the proportion of data per bin
    binned["weight"] = binned_counts / n

    weighed_diff = abs(binned["y_pred"] - binned["y"]) * binned["weight"]
    return weighed_diff.sum()


def sce(y_true: np.array, y_pred: np.array, num_bins: int = 10) -> float:
    """
    Measure the Static Calibration Error (SCE) by [2], an extension to the Expected Calibration Error to multiple
    classes.

    Parameters
    ----------
    y_true: np.array
        True labels for each input.
    y_pred: np.array
        Categorical probability distribution for each input.
    num_bins: int
        Number of bins. Default is 10.

    Returns
    -------
    float
        Static Calibration Error.
    """
    assert len(y_pred.shape) == 2, "y_pred must be a matrix!"
    assert (
        y_true.shape[0] == y_pred.shape[0]
    ), "Shapes of y_true and y_pred do not match!"

    N = len(y_true)
    num_classes = y_pred.shape[1]
    bins = np.arange(0, 1, 1 / num_bins)
    bin_indices = np.digitize(np.max(y_pred, axis=1), bins)
    sce = 0

    for bin in range(num_bins):
        # Get predictions and labels for the current bin
        bin_preds = y_pred[bin_indices == bin, :]
        bin_labels = y_true[bin_indices == bin]

        for k in range(num_classes):
            # Get accuracy and confidence for the current class k in the current bin
            bin_class_preds = bin_preds[bin_labels == k, :]

            if bin_class_preds.shape[0] == 0:
                continue

            n_bk = bin_class_preds.shape[0]
            bin_class_acc = np.mean(
                (np.argmax(bin_class_preds, axis=1) == k).astype(float)
            )
            bin_class_conf = np.mean(np.max(bin_class_preds, axis=1))
            sce += n_bk / N * abs(bin_class_acc - bin_class_conf)

    sce /= num_classes

    return sce


def ace(y_true: np.array, y_pred: np.array, num_ranges: int = 10) -> float:
    """
     Measure the Adaptive Calibration Error (ACE) by [2], an version of the static calibration error that uses ranges
     instead of bins. Every range contains the same number of predictions.

    Parameters
     ----------
     y_true: np.array
         True labels for each input.
     y_pred: np.array
         Categorical probability distribution for each input.
     num_ranges: int
         Number of ranges. Default is 10.

     Returns
     -------
     float
         Adaptive Calibration Error.
    """
    assert len(y_pred.shape) == 2, "y_pred must be a matrix!"
    assert (
        y_true.shape[0] == y_pred.shape[0]
    ), "Shapes of y_true and y_pred do not match!"

    N = len(y_true)
    num_classes = y_pred.shape[1]
    confs = np.sort(np.max(y_pred, axis=1))
    step = int(np.floor(N / num_ranges))  # Inputs per range
    thresholds = np.repeat(
        np.array([confs[i] for i in range(0, step * num_ranges, step)])[np.newaxis, ...], N, axis=0
    )  # Get the thresholds corresponding to ranges

    max_preds = np.repeat(
        np.max(y_pred, axis=1)[..., np.newaxis], num_ranges, axis=1
    )  # Repeat all maximum predictions
    b = (max_preds <= thresholds).astype(
        int
    )  # Compare max predictions against thresholds
    bin_indices = np.argmax(b, axis=1)
    ace = 0

    for bin in range(num_ranges):
        bin_preds = y_pred[bin_indices == bin, :]
        bin_labels = y_true[bin_indices == bin]

        for k in range(num_classes):
            bin_class_preds = bin_preds[bin_labels == k, :]

            if bin_class_preds.shape[0] == 0:
                continue

            bin_class_acc = np.mean(
                (np.argmax(bin_class_preds, axis=1) == k).astype(int)
            )
            bin_class_conf = np.mean(np.max(bin_class_preds, axis=1))
            ace += abs(bin_class_acc - bin_class_conf)

    ace /= num_classes * num_ranges

    return ace


def coverage_percentage(y_true: np.array, y_pred: np.array, alpha: float = 0.05):
    """
    Return the percentage of times the true prediction was contained in the 1 - alpha prediction set. Based on the work
    by [3].

    [3] Kompa, Benjamin, Jasper Snoek, and Andrew L. Beam. "Empirical frequentist coverage of deep learning uncertainty
    quantification procedures." Entropy 23.12 (2021): 1608.

    Parameters
    ----------
    y_true: np.array
         True labels for each input.
    y_pred: np.array
         Categorical probability distribution for each input.
    alpha: float
        Probability mass threshold.

    Returns
    -------
    float
        Coverage percentage.
    """
    sorted_indices = np.argsort(-y_pred, axis=1)  # Add minus to sort descendingly
    # See https://stackoverflow.com/questions/19775831/row-wise-indexing-in-numpy for explanation for expression below
    sorted_probs = y_pred[np.arange(y_pred.shape[0])[:, None], sorted_indices]
    cum_probs = np.cumsum(sorted_probs, axis=1)

    # Create boolean array as int to determine the classes in the prediction set
    thresholded_cum_probs = (cum_probs >= (1 - alpha - 1e-8)).astype(int)

    # Use argmax to find first class for which the 1 - alpha threshold is surpassed - all other classes are outside
    # of the prediction set.
    cut_indices = np.argmax(thresholded_cum_probs, axis=1) + 1

    # Check if class is contained in prediction set
    num_covered = 0
    for i, cut_idx in enumerate(cut_indices):
        num_covered += int(y_true[i] in sorted_indices[i, :cut_idx])

    coverage_percentage = num_covered / y_true.shape[0]

    return coverage_percentage


def coverage_width(y_pred: np.array, alpha: float = 0.05, eps: float = 1e-8, **kwargs):
    """
    Return the width of the 1 - alpha prediction set. Based on the work by [3].

    [3] Kompa, Benjamin, Jasper Snoek, and Andrew L. Beam. "Empirical frequentist coverage of deep learning uncertainty
    quantification procedures." Entropy 23.12 (2021): 1608.

    Parameters
    ----------
    y_pred: np.array
         Categorical probability distribution for each input.
    alpha: float
        Probability mass threshold.
    eps: float
        Small number to avoid floating point precision problems.

    Returns
    -------
    float
        Average prediction set width.
    """
    sorted_indices = np.argsort(-y_pred, axis=1)  # Add minus to sort descendingly
    # See https://stackoverflow.com/questions/19775831/row-wise-indexing-in-numpy for explanation for expression below
    sorted_probs = y_pred[np.arange(y_pred.shape[0])[:, None], sorted_indices]
    cum_probs = np.cumsum(sorted_probs, axis=1)

    # Create boolean array as int to determine the classes in the prediction set
    thresholded_cum_probs = (cum_probs >= (1 - alpha - eps)).astype(int)

    # Use argmax to find first class for which the 1 - alpha threshold is surpassed - all other classes are outside
    # of the prediction set.
    widths = np.argmax(thresholded_cum_probs, axis=1).astype(float) + 1

    # Compute average width
    average_width = np.mean(widths)

    return average_width


def evaluate_calibration(
    model: Model,
    eval_split: DataLoader,
    eval_funcs: Dict[str, Callable] = frozendict({
        "ece": ece,
        "sce": sce,
        "ace": ace,
        "coverage_percentage": coverage_percentage,
        "coverage_width": coverage_width
    }),
    ignore_token_ids: Tuple[int] = (-100, ),
) -> Dict[str, Any]:
    """
    Evaluate the calibration properties of a model.

    Parameters
    ----------
    model: Model
        Model to be evaluated.
    eval_split: DataLoader
        Evaluation split.
    eval_funcs: Dict[str, Callable]
        Evaluation functions used to assess calibration properties.
    ignore_token_ids: Tuple[int]
        IDs of tokens that should be ignored by the model during evaluation.

    Returns
    -------
    Dict[str, Any]
        Results as a dictionary from uncertainty metric / split / eval metric to result.
    """
    # Initialize data structure that track stats
    scores = defaultdict(float)  # Final scores
    sentence_i = 0

    # Get scores for eval split
    split_predictions = []  # Collect all predictions on this split
    split_labels = []  # Collect all labels on this split

    for batch in eval_split:
        attention_mask, input_ids, labels = (
            batch["attention_mask"].to(model.device),
            batch["input_ids"].to(model.device),
            batch["labels"].to(model.device),
        )

        # Determine if sequence labelling / token prediction or sequence predction
        if len(labels.shape) == 2:
            batch_size, seq_len = labels.shape
        else:
            batch_size, seq_len = labels.shape[0], 1

        # Get predictions
        with torch.no_grad():
            predictions = model.predict(input_ids, attention_mask=attention_mask)

        # Reshape for easier processing
        predictions = rearrange(predictions, "b t p -> (b t) p")

        # Filter irrelevant tokens for language modelling / sequence labelling / token predictions
        batch_mask = rearrange(
            torch.all(
                torch.stack([input_ids != idx for idx in ignore_token_ids]), dim=0
            ),
            "b s -> (b s)",
        ).to(model.device)

        # If the task is not sequence classification, we also compute the (mean) sequence loss
        if not model.module.is_sequence_classifier:
            labels = rearrange(labels, "b l -> (b l)")
            predictions = predictions[batch_mask]
            labels = labels[batch_mask]

        split_predictions.append(predictions.detach().cpu().numpy())
        split_labels.append(labels.detach().cpu().numpy())

        sentence_i += batch_size

    # Simply all data structures
    split_predictions = np.concatenate(split_predictions, axis=0)
    split_labels = np.concatenate(split_labels, axis=0)

    # Mask out predictions for -100
    label_mask = split_labels != -100
    split_labels = split_labels[label_mask]
    split_predictions = split_predictions[label_mask]

    # Compute calibration scores
    for name, eval_func in eval_funcs.items():
        scores[name] = eval_func(y_true=split_labels, y_pred=split_predictions)

    return scores
