#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

from losses.dice import DiceLoss


class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5
    ):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        logits,
        targets
    ):

        bce = self.bce(
            logits,
            targets
        )

        dice = self.dice(
            logits,
            targets
        )

        loss = (
            self.bce_weight * bce
            +
            self.dice_weight * dice
        )

        return loss