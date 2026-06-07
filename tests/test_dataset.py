#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:54:47 2026

@author: navdeepkaushish
"""

from datasets.fish_dataset import FishDataset

dataset = FishDataset(
    root_dir="processed_images",
    metadata_csv="metadata/metadata.csv"
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print()

print(
    "Image:",
    sample["image"].shape,
    sample["image"].dtype
)

print(
    "Mask:",
    sample["mask"].shape,
    sample["mask"].dtype
)

print()

print(
    "Mask min:",
    sample["mask"].min()
)

print(
    "Mask max:",
    sample["mask"].max()
)

print()

print(
    sample["image_id"]
)

print(
    sample["fish_id"]
)

print(
    sample["genotype"]
)

dataset = FishDataset(
    root_dir="processed_images",
    metadata_csv="metadata/metadata.csv",
    patch_size=512
)

sample = dataset[0]

print(sample["image"].shape)
print(sample["mask"].shape)
print(sample["mask"].sum())

sample = dataset[0]

print(sample["mask"].sum())

for c in range(25):
    print(
        c,
        sample["mask"][c].sum().item()
    )