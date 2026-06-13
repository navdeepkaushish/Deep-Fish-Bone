#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:19:58 2026

@author: navdeepkaushish
"""

import torch

from datasets.metadata import Metadata
from datasets.splitter import FishSplitter

from engine.dataloaders import (
    build_full_valid_loader
)

from engine.validate_full import (
    validate_full_image
)

from models.unetpp import (
    build_model
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

meta = Metadata(
    "metadata/metadata.csv"
)

df = meta.dataframe()

splitter = FishSplitter(
    n_splits=5
)

fold, train_idx, valid_idx = next(
    splitter.split(df)
)

loader = build_full_valid_loader(

    root_dir="ventral",

    metadata_csv=
    "metadata/metadata.csv",

    valid_idx=valid_idx

)

model = build_model().to(device)

mean_dice, class_dice, counts = (

    validate_full_image(

        model,

        loader,

        device

    )

)

print()

print(
    "Mean Dice:",
    mean_dice
)

print()

for d, c in zip(
    class_dice,
    counts
):
    print(d, c)