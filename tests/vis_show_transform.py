#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:13:17 2026

@author: navdeepkaushish
"""

from datasets.fish_dataset import FishDataset
from datasets.transforms import get_train_transforms
from tests.show_sample import show_sample

dataset = FishDataset(
    root_dir="processed_images",
    metadata_csv="metadata/metadata.csv",
    patch_size=512,
    transforms=get_train_transforms()
)

for i in range(5):

    sample = dataset[5]

    show_sample(sample)