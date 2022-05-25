"""
Define some uncertainty metrics for neural discriminators. These usually operate on the predicted logits unless
specified otherwise.
"""

# EXT
import torch


def max_prob(logits: torch.FloatTensor) -> torch.FloatTensor:
    """
    Compute the maximum softmax probability baseline by [1] for a tensor of batch_size x seq_len x output_size.
    Because we want a high value when uncertainty is high, we actually compute 1 - max. prob.

    [1] https://arxiv.org/pdf/1610.02136.pdf

    Parameters
    ----------
    logits: torch.FloatTensor
        Logits of the current batch.

    Returns
    -------
    torch.FloatTensor
        Max. prob. values for the current batch.
    """
    probs = torch.softmax(logits, dim=-1)
    max_prob = 1 - torch.max(probs, dim=-1)[0]

    return max_prob


def softmax_gap(logits: torch.FloatTensor) -> torch.FloatTensor:
    """
    Compute softmax gap by [2] for a tensor of batch_size x seq_len x output_size.

    [2] https://arxiv.org/pdf/1811.00908.pdf

    Parameters
    ----------
    logits: torch.FloatTensor
        Logits of the current batch.

    Returns
    -------
    torch.FloatTensor
        Softmax gap.
    """
    probs = torch.softmax(logits, dim=-1)
    max_prob, max_idx = torch.max(probs, dim=-1)
    probs[:, :, max_idx] = 0
    gap = 1 - (max_prob - torch.max(probs, dim=-1)[0])

    return gap


def predictive_entropy(logits: torch.FloatTensor, eps: float = 1e-5) -> torch.FloatTensor:
    """
    Compute predictive entropy for a tensor of batch_size x seq_len x output_size.

    Parameters
    ----------
    logits: torch.FloatTensor
        Logits of the current batch.

    Returns
    -------
    torch.FloatTensor
        Predictive entropy for the current batch.
    """
    probs = torch.softmax(logits, dim=-1)
    pred_entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)

    return pred_entropy


def dempster_shafer(logits: torch.FloatTensor) -> torch.FloatTensor:
    """
    Compute the dempster-shafer metric [2] for a tensor of batch_size x seq_len x output_size.

    [2] https://proceedings.neurips.cc/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf

    Parameters
    ----------
    logits: torch.FloatTensor
        Logits of the current batch.

    Returns
    -------
    torch.FloatTensor
        Dempster-shafer metric for the current batch.
    """
    num_classes = logits.shape[2]

    return num_classes / (num_classes + torch.exp(logits).sum(dim=-1))


def variance(logits: torch.FloatTensor) -> torch.FloatTensor:
    """
    Compute the variance in predictions given a number of predictions. Thus, this metric expects
    a logit tensor of size batch_size x num_predictions x seq_len x output_size.

    Parameters
    ----------
    logits: torch.FloatTensor
        Logits of the current batch.

    Returns
    -------
    torch.FloatTensor
        Variance in predictions for the current batch.
    """
    probs = torch.softmax(logits, dim=-1)
    var = torch.var(probs, dim=1).mean(dim=-1)

    return var


def mutual_information(logits: torch.FloatTensor, eps: float = 1e-5) -> torch.FloatTensor:
    """
    Compute the mutual information as defined in [3] given a number of predictions. Thus, this metric expects
    a logit tensor of size batch_size x num_predictions x seq_len x output_size.

    [3] https://arxiv.org/pdf/1803.08533.pdf

    Parameters
    ----------
    logits: torch.FloatTensor
        Logits of the current batch.

    Returns
    -------
    torch.FloatTensor
       Mutual information for the current batch.
    """
    probs = torch.softmax(logits, dim=-1)
    mutual_info = -(probs.mean(dim=1) * torch.log(probs.mean(dim=1) + eps)).sum(dim=-1) + (
        probs * torch.log(probs + eps)
    ).sum(dim=-1).mean(dim=1)

    return mutual_info
