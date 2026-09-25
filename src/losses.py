"""Losses for detector training."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HeatmapLoss(nn.Module):
    """BCE with logits plus soft Dice loss."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)

        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        intersection = torch.sum(prob * target, dim=dims)
        denominator = torch.sum(prob + target, dim=dims)
        dice = 1.0 - torch.mean((2.0 * intersection + 1.0) / (denominator + 1.0))

        return self.bce_weight * bce + self.dice_weight * dice
