#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:34:25 2026

@author: navdeepkaushish
"""

from datasets.fish_dataset import FishDataset
from datasets.transforms import get_train_transforms

dataset = FishDataset(
    root_dir="processed_images",
    metadata_csv="metadata/metadata.csv",
    patch_size=512,
    transforms=get_train_transforms()
)

sample = dataset[0]

print(sample["image"].shape)
print(sample["mask"].shape)

print(
    sample["image"].min(),
    sample["image"].max()
)

print(
    sample["mask"].min(),
    sample["mask"].max()
)