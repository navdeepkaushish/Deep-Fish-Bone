#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 09:59:36 2026

@author: navdeepkaushish
"""

from datasets.fish_dataset import FishDataset
#==== image mode =============================================================
ds = FishDataset(
    root_dir="ventral",
    metadata_csv="metadata/metadata.csv",
    patch_size=None,
    resize_to=(966, 1288)
)

s = ds[0]
print(s["image"].shape)  # torch.Size([3, 966, 1288])
print(s["mask"].shape)   # torch.Size([25, 966, 1288])
#=================== Patch Mode ==============================================
ds = FishDataset(
    root_dir="ventral",
    metadata_csv="metadata/metadata.csv",
    patch_size=512,
    resize_to=(966, 1288)
)

s = ds[0]
print(s["image"].shape)  # torch.Size([3, 512, 512])
print(s["mask"].shape)   # torch.Size([25, 512, 512])