#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:56:20 2026

@author: navdeepkaushish
"""

import torch.nn as nn

from losses.dice import DiceLoss


class CombinedLoss(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

        self.dice = DiceLoss()

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
            0.5 * bce
            +
            0.5 * dice
        )

        return loss