#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 09:56:53 2026

@author: navdeepkaushish
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from datasets.patch_sampler import PatchSampler


MASK_NAMES = [
    "Br1a", "Br1b", "Br2a", "Br2b",
    "CB1", "CB2",
    "CH1", "CH2",
    "CL1", "CL2",
    "D1", "D2",
    "EN1", "EN2",
    "Hm1", "Hm2",
    "M1", "M2",
    "N",
    "Oc1", "Oc2",
    "Op1", "Op2",
    "P",
    "VC"
]


class FishDataset(Dataset):

    def __init__(
        self,
        root_dir,
        metadata_csv,
        indices=None,
        patch_size=None,
        resize_to=(966, 1288),   # (H, W)
        transforms=None
    ):

        self.root_dir = Path(root_dir)
        self.meta = pd.read_csv(metadata_csv)

        if indices is not None:
            self.meta = self.meta.iloc[indices]
            self.meta = self.meta.reset_index(drop=True)

        self.resize_to = resize_to

        self.patch_sampler = None

        if patch_size is not None:
            self.patch_sampler = PatchSampler(
                patch_size=patch_size
            )

        self.transforms = transforms

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

    def resize_image_and_masks(self, image, masks):

        if self.resize_to is None:
            return image, masks

        target_h, target_w = self.resize_to

        image = cv2.resize(
            image,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )

        resized_masks = []

        for m in masks:

            m_resized = cv2.resize(
                m,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            )

            m_resized = (m_resized > 0).astype(np.float32)

            resized_masks.append(m_resized)

        masks = np.stack(
            resized_masks,
            axis=0
        )

        return image, masks

    def __getitem__(self, idx):

        row = self.meta.iloc[idx]

        sample_dir = self.root_dir / str(row["image_id"])

        image = np.array(
            Image.open(
                sample_dir / "image.png"
            ).convert("RGB"),
            dtype=np.float32
        )

        image = image / 255.0

        H, W = image.shape[:2]

        masks = []

        for name in MASK_NAMES:

            mask = self.load_binary_mask(
                sample_dir / f"{name}.png"
            )

            if mask.shape != (H, W):
                raise RuntimeError(
                    f"{name}.png size mismatch "
                    f"{mask.shape} vs {(H, W)}"
                )

            masks.append(mask)

        masks = np.stack(
            masks,
            axis=0
        )

        # Resize before patch extraction
        image, masks = self.resize_image_and_masks(
            image,
            masks
        )

        # Convert image from HWC to CHW
        image = image.transpose(2, 0, 1)

        # Patch extraction
        if self.patch_sampler is not None:

            image, masks = self.patch_sampler.sample(
                image,
                masks
            )

        # Albumentations
        if self.transforms is not None:

            transformed = self.transforms(
                image=image.transpose(1, 2, 0),
                mask=masks.transpose(1, 2, 0)
            )

            image = transformed["image"]
            masks = transformed["mask"]

            image = image.transpose(2, 0, 1)
            masks = masks.transpose(2, 0, 1)

        image = torch.from_numpy(
            image.astype(np.float32)
        )

        masks = torch.from_numpy(
            masks.astype(np.float32)
        )

        return {
            "image": image,
            "mask": masks,
            "image_id": str(row["image_id"]),
            "fish_id": str(row["fish_id"]),
            "genotype": str(row["genotype"]),
            "quality": int(row["quality"])
        }