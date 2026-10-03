#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Final class-specific Br2 loss.

Same 0.5 regional / 0.5 focal weighting as CombinedLoss.
Dice is used for all structures except Br2a and Br2b, where
Tversky loss (alpha=0.7, beta=0.3 by default) replaces Dice.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.structure_names import STRUCTURES
from losses.combined import CombinedLoss


BR2_IDXS = [STRUCTURES.index("Br2a"), STRUCTURES.index("Br2b")]


class ClassSpecificBr2Loss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

        combined = CombinedLoss()

        if not hasattr(combined, "focal"):
            raise RuntimeError(
                "CombinedLoss does not expose `.focal`; refusing to substitute "
                "a different focal-loss implementation."
            )

        self.focal = combined.focal
        self.region_weight = float(combined.dice_weight)
        self.focal_weight = float(combined.focal_weight)
        self.focal_source = "existing_combined_loss"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        targets = targets.float()
        dims = (0, 2, 3)

        tp = (probabilities * targets).sum(dim=dims)
        fp = (probabilities * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probabilities) * targets).sum(dim=dims)

        dice_score = (
            (2.0 * tp + self.smooth)
            / (2.0 * tp + fp + fn + self.smooth)
        )
        per_structure_region_loss = 1.0 - dice_score

        for idx in BR2_IDXS:
            tversky_score = (
                (tp[idx] + self.smooth)
                / (
                    tp[idx]
                    + self.alpha * fp[idx]
                    + self.beta * fn[idx]
                    + self.smooth
                )
            )
            per_structure_region_loss[idx] = 1.0 - tversky_score

        region_loss = per_structure_region_loss.mean()
        focal_loss = self.focal(logits, targets)

        return (
            self.region_weight * region_loss
            + self.focal_weight * focal_loss
        )
