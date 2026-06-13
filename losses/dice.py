#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:53:46 2026

@author: navdeepkaushish
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1e-6
    ):
        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        probs = torch.sigmoid(
            logits
        )

        probs = probs.reshape(
            probs.shape[0],
            probs.shape[1],
            -1
        )

        targets = targets.reshape(
            targets.shape[0],
            targets.shape[1],
            -1
        )

        intersection = (
            probs * targets
        ).sum(-1)

        union = (
            probs.sum(-1)
            +
            targets.sum(-1)
        )

        dice = (

            2 * intersection
            + self.smooth

        ) / (

            union
            + self.smooth

        )

        loss = 1 - dice

        return loss.mean()