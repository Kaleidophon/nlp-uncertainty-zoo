"""
Module to implement different metrics the quality of uncertainty metrics.
"""

# EXT
import numpy as np
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
    return kendalltau(losses, uncertainties)[0]


def cvmc(uncertainties_a: np.array, uncertainties_b: np.array) -> float:
    """
    Compute the Cramer-van Mises criterion between two empirical cumulative distributions functions, using the
    rank-based approximation by [1].

    [1] https://arxiv.org/pdf/1802.06332.pdf

    Parameters
    ----------
    uncertainties_a: np.array
        Uncertainty scores of model A.
    uncertainties_b: np.array
        Uncertainty scores of model B.

    Returns
    -------
    float:
        Value of the Cramer-van Mises criterion.
    """
    m = len(uncertainties_a)
    n = len(uncertainties_b)
    # Add small value so that ranks of same scores are different when they appear in the two different scores
    uncertainties_b += 1e-13
    combined = np.concatenate([uncertainties_a, uncertainties_b])
    combined = np.sort(combined)
    rank = {score: rank for rank, score in enumerate(combined)}

    mm, mn, nn = 0, 0, 0

    for x in uncertainties_a:
        for xx in uncertainties_a:
            mm += abs(rank[x] - rank[xx])

    for x in uncertainties_a:
        for y in uncertainties_b:
            mn += abs(rank[x] - rank[y])

    for y in uncertainties_b:
        for yy in uncertainties_b:
            nn += abs(rank[y] - rank[yy])

    cvmc = m * n / (m + n) * (mn / (m * n) - mm / (2 * m ** 2) - nn / (2 * nn * 2))

    return cvmc


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


def ace(y_true: np.array, y_pred: np.array, num_ranges: int = 4) -> float:
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
        np.array([confs[i] for i in range(0, N, step)])[np.newaxis, ...], N, axis=0
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
