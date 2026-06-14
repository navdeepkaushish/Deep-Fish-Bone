#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from losses.focal import FocalLoss
from losses.combined import CombinedLoss


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

focal = FocalLoss()
combined = CombinedLoss()

focal_loss = focal(
    logits,
    targets
)

combined_loss = combined(
    logits,
    targets
)

print(
    "Focal loss:",
    focal_loss.item()
)

print(
    "Combined loss:",
    combined_loss.item()
)

assert torch.isfinite(
    focal_loss
)

assert torch.isfinite(
    combined_loss
)

print(
    "Focal/Combined loss test passed."
)