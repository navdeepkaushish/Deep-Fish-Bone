#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

from losses.dice import DiceLoss
from losses.focal import FocalLoss


class CombinedLoss(nn.Module):

    def __init__(
        self,
        dice_weight=0.5,
        focal_weight=0.5
    ):
        super().__init__()

        self.dice = DiceLoss()

        self.focal = FocalLoss(
            alpha=0.25,
            gamma=2.0
        )

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(
        self,
        logits,
        targets
    ):

        dice = self.dice(
            logits,
            targets
        )

        focal = self.focal(
            logits,
            targets
        )

        loss = (
            self.dice_weight * dice
            +
            self.focal_weight * focal
        )

        return loss