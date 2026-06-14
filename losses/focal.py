#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-label focal loss for sigmoid outputs.

    logits:  (B, C, H, W)
    targets: (B, C, H, W), values 0/1
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        reduction="mean"
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits,
        targets
    ):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(
            logits
        )

        pt = torch.where(
            targets == 1,
            probs,
            1 - probs
        )

        focal_weight = (
            (1 - pt) ** self.gamma
        )

        alpha_weight = torch.where(
            targets == 1,
            self.alpha,
            1 - self.alpha
        )

        loss = (
            alpha_weight
            *
            focal_weight
            *
            bce
        )

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss