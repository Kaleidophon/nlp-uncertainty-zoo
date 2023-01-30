"""
Module to implement different metrics the quality of uncertainty metrics.
"""

# STD
from typing import Dict, Any, Optional, Tuple, Callable

# EXT
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import kendalltau
from torch.utils.data import DataLoader

# PROJECT
from nlp_uncertainty_zoo.models.model import Model


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


def evaluate_uncertainty(
    model: Model,
    eval_split: DataLoader,
    ood_eval_split: Optional[DataLoader] = None,
    eval_func: Tuple[Callable] = (kendalltau,),
    contrastive_eval_func: Tuple[Callable] = (aupr, auroc)
) -> Dict[str, Any]:
    """
    Evaluate the uncertainty properties of a model. Evaluation happens in two ways:

        1. Eval functions that are applied to uncertainty metrics of the model on the `eval_split` (and `ood_eval_split`
        if specified).
        2. Eval functions that take measurements on and in- and out-of-distribution dataset to evaluate a proxy binary
        anomaly detection task, for which the functions specified by `contrastive_eval_func` are used. Also, the
        `ood_eval_split` argument has to be specified.

    Parameters
    ----------
    model: Model
        Model to be evaluated.
    eval_split: DataLoader
        Main evaluation split.
    ood_eval_split: Optional[DataLoader]
        OOD evaluation split. Needs to be specified for contrastive evalualtion functions to work.
    eval_func
    contrastive_eval_func

    Returns
    -------

    """
    ... # TODO
