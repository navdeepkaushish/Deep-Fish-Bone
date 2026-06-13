#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:00:41 2026

@author: navdeepkaushish
"""

import torch


def multiclass_dice(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6
):

    probs = torch.sigmoid(logits)

    preds = (probs > threshold).float()

    preds = preds.reshape(
        preds.shape[0],
        preds.shape[1],
        -1
    )

    targets = targets.reshape(
        targets.shape[0],
        targets.shape[1],
        -1
    )

    intersection = (
        preds * targets
    ).sum(-1)

    union = (
        preds.sum(-1)
        +
        targets.sum(-1)
    )

    dice = (

        2 * intersection
        + smooth

    ) / (

        union
        + smooth

    )

    return dice.mean().item()