from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    loss_type: str,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
):
    loss_type = str(loss_type).lower()

    if loss_type == "ce":
        def _loss(logits, targets):
            return F.cross_entropy(logits, targets, weight=class_weights)
        return _loss

    if loss_type == "focal":
        focal = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            reduction="mean",
        )
        return focal

    raise ValueError(f"Unsupported loss_type: {loss_type}")