#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:51:51 2026

@author: navdeepkaushish
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from datasets.patch_sampler import PatchSampler


MASK_NAMES = [
    "Br1a",
    "Br1b",
    "Br2a",
    "Br2b",
    "CB1",
    "CB2",
    "CH1",
    "CH2",
    "CL1",
    "CL2",
    "D1",
    "D2",
    "EN1",
    "EN2",
    "Hm1",
    "Hm2",
    "M1",
    "M2",
    "N",
    "Oc1",
    "Oc2",
    "Op1",
    "Op2",
    "P",
    "VC"
]


class FishDataset(Dataset):

    def __init__(
        self,
        root_dir,
        metadata_csv,
        indices=None,
        patch_size=None
    ):

        self.root_dir = Path(root_dir)

        self.meta = pd.read_csv(metadata_csv)

        if indices is not None:
            self.meta = self.meta.iloc[indices]
            self.meta = self.meta.reset_index(drop=True)
            
        self.patch_sampler = None

        if patch_size is not None:
            self.patch_sampler = PatchSampler(
            patch_size=patch_size
        )

    def __len__(self):
        return len(self.meta)

    @staticmethod
    def load_binary_mask(mask_path):

        mask = np.array(
            Image.open(mask_path).convert("L"),
            dtype=np.uint8
        )

        mask = (mask > 0).astype(np.float32)

        return mask

    def __getitem__(self, idx):

        row = self.meta.iloc[idx]

        sample_dir = self.root_dir / str(row["image_id"])

        # ------------------------
        # RGB image
        # ------------------------

        image = np.array(
            Image.open(
                sample_dir / "image.png"
            ).convert("RGB"),
            dtype=np.float32
        )

        image = image / 255.0

        H, W = image.shape[:2]

        # ------------------------
        # masks
        # ------------------------

        masks = []

        for name in MASK_NAMES:

            mask = self.load_binary_mask(
                sample_dir / f"{name}.png"
            )

            if mask.shape != (H, W):
                raise RuntimeError(
                    f"{name}.png size mismatch "
                    f"{mask.shape} vs {(H,W)}"
                )

            masks.append(mask)

        masks = np.stack(
            masks,
            axis=0
        )

        image = image.transpose(2, 0, 1)
        
        if self.patch_sampler is not None:

            image, masks = self.patch_sampler.sample(
                image,
                masks
                )
            
        image = torch.from_numpy(
            image.astype(np.float32)
            )

        masks = torch.from_numpy(
            masks.astype(np.float32)
            )



        return {
            "image": image,
            "mask": masks,
            "image_id": row["image_id"],
            "fish_id": row["fish_id"],
            "genotype": row["genotype"],
            "quality": row["quality"]
            }