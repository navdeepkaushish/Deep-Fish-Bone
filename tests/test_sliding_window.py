#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:10:36 2026

@author: navdeepkaushish
"""

import torch

from datasets.metadata import Metadata
from datasets.splitter import FishSplitter
from engine.dataloaders import build_full_valid_loader
from models.unetpp import build_model
from inference.sliding_window import sliding_window_predict


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

meta = Metadata("metadata/metadata.csv")
df = meta.dataframe()

splitter = FishSplitter(n_splits=5)
fold, train_idx, valid_idx = next(splitter.split(df))

loader = build_full_valid_loader(
    root_dir="ventral",
    metadata_csv="metadata/metadata.csv",
    valid_idx=valid_idx
)

batch = next(iter(loader))

image = batch["image"][0]

model = build_model().to(device)

probs = sliding_window_predict(
    model=model,
    image=image,
    device=device,
    patch_size=512,
    stride=256
)

print(probs.shape)
print(probs.min().item(), probs.max().item())