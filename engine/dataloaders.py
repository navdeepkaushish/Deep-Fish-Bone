#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DataLoaders for patch training and full-image validation/test evaluation."""

import torch
from torch.utils.data import DataLoader

from datasets.fish_dataset import FishDataset
from datasets.transforms import get_train_transforms, get_valid_transforms


def _build_dataset(
    root_dir,
    metadata_csv,
    indices,
    patch_size,
    resize_to,
    training,
):
    return FishDataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        indices=indices,
        patch_size=patch_size,
        resize_to=resize_to,
        transforms=(
            get_train_transforms()
            if training
            else get_valid_transforms()
        ),
    )


def build_patch_loaders(
    root_dir,
    metadata_csv,
    train_idx,
    val_idx,
    batch_size=4,
    patch_size=512,
    resize_to=(966, 1288),
    num_workers=0,
):
    """
    Patch-based training and patch-based inner validation.

    Patch validation is for diagnostic monitoring only. Model selection must
    use the separate full-image inner-validation loader.
    """
    train_dataset = _build_dataset(
        root_dir, metadata_csv, train_idx, patch_size, resize_to, True
    )
    val_dataset = _build_dataset(
        root_dir, metadata_csv, val_idx, patch_size, resize_to, False
    )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def build_full_image_loader(
    root_dir,
    metadata_csv,
    indices,
    resize_to=(966, 1288),
    num_workers=0,
):
    """Build a batch-size-one loader containing complete resized images."""
    dataset = _build_dataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        indices=indices,
        patch_size=None,
        resize_to=resize_to,
        training=False,
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# Backward-compatible aliases.
def build_dataloaders(*args, **kwargs):
    return build_patch_loaders(*args, **kwargs)


def build_full_valid_loader(
    root_dir,
    metadata_csv,
    valid_idx,
    resize_to=(966, 1288),
    num_workers=0,
):
    return build_full_image_loader(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        indices=valid_idx,
        resize_to=resize_to,
        num_workers=num_workers,
    )
