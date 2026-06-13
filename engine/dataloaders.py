#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:37:50 2026

@author: navdeepkaushish
"""

import torch
from torch.utils.data import DataLoader

from datasets.fish_dataset import FishDataset
from datasets.transforms import (
    get_train_transforms,
    get_valid_transforms
)


def build_dataloaders(
    root_dir,
    metadata_csv,
    train_idx,
    valid_idx,
    batch_size=4,
    patch_size=512,
    resize_to=(966, 1288),
    num_workers=0
):
    train_dataset = FishDataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        indices=train_idx,
        patch_size=patch_size,
        resize_to=resize_to,
        transforms=get_train_transforms()
    )

    # patch-based validation, useful for quick monitoring only
    valid_dataset = FishDataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        indices=valid_idx,
        patch_size=patch_size,
        resize_to=resize_to,
        transforms=get_valid_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, valid_loader


def build_full_valid_loader(
    root_dir,
    metadata_csv,
    valid_idx,
    resize_to=(966, 1288),
    num_workers=0
):
    valid_dataset = FishDataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        indices=valid_idx,
        patch_size=None,          # IMPORTANT: full image
        resize_to=resize_to,
        transforms=get_valid_transforms()
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,             # IMPORTANT: full image validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return valid_loader