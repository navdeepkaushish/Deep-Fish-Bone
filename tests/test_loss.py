#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:57:23 2026

@author: navdeepkaushish
"""

import torch

from losses.combined import (
    CombinedLoss
)

criterion = CombinedLoss()

logits = torch.randn(
    2,
    25,
    512,
    512
)

targets = torch.randint(
    0,
    2,
    (
        2,
        25,
        512,
        512
    )
).float()

loss = criterion(
    logits,
    targets
)

print(loss)