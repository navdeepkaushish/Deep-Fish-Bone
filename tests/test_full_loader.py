#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 10:55:48 2026

@author: navdeepkaushish
"""

from datasets.metadata import Metadata
from datasets.splitter import FishSplitter
import matplotlib.pyplot as plt

from engine.dataloaders import (
    build_full_valid_loader
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

batch = next(iter(loader))

print(
    batch["image"].shape
)

print(
    batch["mask"].shape
)

print(
    batch["image_id"][0]
)

print(
    batch["fish_id"][0]
)

print(
    batch["mask"].sum()
)

#========== Visualize ===========================
import matplotlib.pyplot as plt

img = batch["image"][0]

plt.imshow(
    img.permute(1,2,0)
)

plt.show()