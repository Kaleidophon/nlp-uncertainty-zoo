"""
Module to implement different metrics the quality of uncertainty metrics.
"""

# STD
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, Callable

# EXT
from einops import rearrange
from frozendict import frozendict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import kendalltau
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    id_eval_split: DataLoader,
    ood_eval_split: Optional[DataLoader] = None,
    eval_funcs: Dict[str, Callable] = frozendict({"kendalls_tau": kendalltau}),
    contrastive_eval_funcs: Tuple[Callable] = frozendict({"aupr": aupr, "auroc": auroc}),
    ignore_token_ids: Tuple[int] = (-100, ),
    verbose: bool = True,
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
    id_eval_split: DataLoader
        Main evaluation split.
    ood_eval_split: Optional[DataLoader]
        OOD evaluation split. Needs to be specified for contrastive evalualtion functions to work.
    eval_funcs: Dict[str, Callable]
        Evaluation function that evaluate uncertainty by comparing it to model losses on a single split.
    contrastive_eval_funcs: Dict[str, Callable]
        Evaluation functions that evaluate uncertainty by comparing uncertainties on an ID and OOD test set.
    ignore_token_ids: Tuple[int]
        IDs of tokens that should be ignored by the model during evaluation.
    verbose: bool
        Whether to display information about the current progress.

    Returns
    -------
    Dict[str, Any]
        Results as a dictionary from uncertainty metric / split / eval metric to result.
    """
    num_batches = len(id_eval_split)

    if ood_eval_split is not None:
        num_batches += len(ood_eval_split)

    progress_bar = tqdm(total=num_batches if verbose else None)

    model_uncertainty_metrics = list(model.available_uncertainty_metrics)
    loss_func = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    # Initialize data structure that track stats
    scores = defaultdict(float)  # Final scores
    # Uncertainties for tokens and sequences (in-distribution)
    id_uncertainties, id_seq_uncertainties = (
        defaultdict(list),
        defaultdict(list)
    )
    # Uncertainties for tokens and sequences (out-of-distribution)
    ood_uncertainties, ood_seq_uncertainties = (
        defaultdict(list),
        defaultdict(list)
    )

    # Initialize result df that will later be written to .csv
    sentence_i = 0

    # Get scores for both test splits
    for (
            split_name,
            eval_split,
            uncertainties,
            seq_uncertainties
    ) in [
        (
                "id",
                id_eval_split,
                id_uncertainties,
                id_seq_uncertainties
        ),
        (
                "ood",
                ood_eval_split,
                ood_uncertainties,
                ood_seq_uncertainties
        ),
    ]:
        split_labels = []  # Collect all labels on this split
        split_losses = []  # Collect all (token) losses on this split
        split_seq_losses = []  # Collect all sequence losses on this split

        for i, batch in enumerate(eval_split):
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
            seq_batch_mask = rearrange(batch_mask, "(b s) -> b s", b=batch_size).to(
                model.device
            )

            # If the task is not sequence classification, we also compute the (mean) sequence loss
            if not model.module.is_sequence_classifier:
                labels = rearrange(labels, "b l -> (b l)")

                # Mask out losses for ignore tokens and recompute sequence losses
                seq_losses = rearrange(
                    loss_func(predictions, labels), "(b l) -> b l", b=batch_size
                )
                seq_losses *= (
                    seq_batch_mask.int()
                )  # Mask out all uninteresting tokens' uncertainties
                seq_losses = seq_losses.mean(dim=1)
                seq_losses *= seq_len
                seq_losses /= seq_batch_mask.int().sum(dim=1)
                split_seq_losses.append(seq_losses.detach().cpu().numpy())

                predictions = predictions[batch_mask]
                labels = labels[batch_mask]

            else:
                seq_losses = loss_func(predictions, labels)
                split_seq_losses.append(seq_losses.detach().cpu().numpy())

            split_labels.append(labels.detach().cpu().numpy())

            # Compute uncertainty
            losses = loss_func(predictions, labels)
            split_losses.append(losses.detach().cpu().numpy())

            for metric_name in model_uncertainty_metrics:
                with torch.no_grad():
                    uncertainty = model.get_uncertainty(
                        input_ids,
                        metric_name=metric_name,
                        attention_mask=attention_mask,
                    )

                seq_uncertainty = torch.clone(uncertainty)
                uncertainty = rearrange(uncertainty, "b l -> (b l)")

                # Filter uncertainties for uninteresting tokens
                if not model.module.is_sequence_classifier:
                    uncertainty = uncertainty[batch_mask]

                    # Get the sequence uncertainties setting non batch-mask tokens to zero and re-normalizing means
                    # across second axis
                    seq_uncertainty *= (
                        seq_batch_mask.int()
                    )  # Mask out all uninteresting tokens' uncertainties
                    seq_uncertainty = seq_uncertainty.mean(dim=1)
                    seq_uncertainty *= seq_len
                    seq_uncertainty /= seq_batch_mask.int().sum(dim=1)
                    seq_uncertainties[metric_name].append(
                        seq_uncertainty.detach().cpu().numpy()
                    )

                    uncertainties[metric_name].append(
                        uncertainty.detach().cpu().numpy()
                    )

                # Sequence classification tasks, sequence uncertainties are just the uncertainties of the single
                # sequence-wide prediction
                else:
                    seq_uncertainties[metric_name].append(
                        uncertainty.detach().cpu().numpy()
                    )

            sentence_i += batch_size

            if verbose:
                progress_bar.set_description(f"Evaluating batch {i + 1}/{num_batches}...")
                progress_bar.update(1)

        # Simply all data structures
        split_losses = np.concatenate(split_losses, axis=0)
        split_seq_losses = np.concatenate(split_seq_losses, axis=0)

        # Compute Kendall's tau scores
        for metric_name in model_uncertainty_metrics:

            for eval_name, eval_func in eval_funcs.items():

                if not model.module.is_sequence_classifier:
                    uncertainties[metric_name] = np.concatenate(uncertainties[metric_name])

                    scores[f"{eval_name}_{split_name}_{metric_name}_token"] = eval_func(
                        split_losses, uncertainties[metric_name]
                    )

                seq_uncertainties[metric_name] = np.concatenate(
                    seq_uncertainties[metric_name]
                )
                scores[f"{eval_name}_{split_name}_{metric_name}_seq"] = eval_func(
                    split_seq_losses, seq_uncertainties[metric_name]
                )

        del split_losses, split_labels

    metric_key = model_uncertainty_metrics[0]
    num_id = len(id_seq_uncertainties[metric_key])
    num_ood = len(ood_seq_uncertainties[metric_key])

    for metric_name in model_uncertainty_metrics:
        for eval_name, eval_func in contrastive_eval_funcs.items():
            scores[f"{eval_name}_{metric_name}"] = eval_func(
                [0] * num_id + [1] * num_ood,
                np.concatenate(
                    (id_seq_uncertainties[metric_name], ood_seq_uncertainties[metric_name])
                ),
            )

    return scores
