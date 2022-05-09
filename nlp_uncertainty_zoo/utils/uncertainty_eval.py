"""
Module to implement different metrics the quality of uncertainty metrics.
"""

# EXT
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import kendalltau


def aupr(y_true: np.array, y_pred: np.array) -> float:
    """
    Return the area under the precision-recall curve for a pseudo binary classification task, where in- and
    out-of-distribution samples correspond to two different classes, which are differentiated using uncertainty scores.

    Parameters
    ----------
    y_true: np.array
        True labels, where 1 corresponds to in- and 0 to out-of-distribution.
    y_pred: np.array
        Uncertainty scores as predictions to distinguish the two classes.

    Returns
    -------
    float
        Area under the precision-recall curve.
    """
    return average_precision_score(y_true, y_pred)


def auroc(y_true: np.array, y_pred: np.array) -> float:
    """
    Return the area under the receiver-operator characteristic for a pseudo binary classification task, where in- and
    out-of-distribution samples correspond to two different classes, which are differentiated using uncertainty scores.

    Parameters
    ----------
    y_true: np.array
        True labels, where 1 corresponds to in- and 0 to out-of-distribution.
    y_pred: np.array
        Uncertainty scores as predictions to distinguish the two classes.

    Returns
    -------
    float
        Area under the receiver-operator characteristic.
    """
    return roc_auc_score(y_true, y_pred)


def kendalls_tau(losses: np.array, uncertainties: np.array) -> float:
    """
    Compute Kendall's tau for a list of losses and uncertainties for a set of inputs. If the two lists are concordant,
    i.e. the points with the highest uncertainty incur the highest loss, Kendall's tau is 1. If they are completely
    discordant, it is -1.

    Parameters
    ----------
    losses: np.array
        List of losses for a set of points.
    uncertainties: np.array
        List of uncertainty for a set of points.

    Returns
    -------
    float
        Kendall's tau, between -1 and 1.
    """
    return kendalltau(losses, uncertainties, nan_policy="omit", method="asymptotic")[0]


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


def coverage_width(y_pred: np.array, alpha: float = 0.05, eps: float = 1e-8):
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
