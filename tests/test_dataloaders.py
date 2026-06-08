#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:39:43 2026

@author: navdeepkaushish
"""

from datasets.metadata import Metadata
from datasets.splitter import FishSplitter

from engine.dataloaders import build_dataloaders


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

train_loader, valid_loader = build_dataloaders(
    root_dir="ventral",
    metadata_csv="metadata/metadata.csv",
    train_idx=train_idx,
    valid_idx=valid_idx,
    batch_size=2,
    patch_size=512,
    num_workers=0   # keep 0 on Mac
)

batch = next(iter(train_loader))

print(batch["image"].shape)
print(batch["mask"].shape)

print(batch["image_id"])
print(batch["fish_id"])