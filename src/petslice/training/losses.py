"""Focal losses for binary (sigmoid) and multi-class (softmax) outputs."""

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25
) -> torch.Tensor:
    if logits.shape[-1] == 1:
        targets = labels.float()
        logits = logits.squeeze(-1)
        cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_true = probabilities * targets + (1 - probabilities) * (1 - targets)
        weights = alpha * targets + (1 - alpha) * (1 - targets)
        return (weights * (1 - p_true) ** gamma * cross_entropy).mean()
    cross_entropy = F.cross_entropy(logits, labels, reduction="none")
    p_true = torch.exp(-cross_entropy)
    return ((1 - p_true) ** gamma * cross_entropy).mean()
