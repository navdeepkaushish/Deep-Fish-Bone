#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:47:23 2026

@author: navdeepkaushish
"""

import random
import numpy as np


class PatchSampler:

    def __init__(
        self,
        patch_size=512,
        foreground_prob=0.9
    ):
        self.patch_size = patch_size
        self.foreground_prob = foreground_prob

    def sample(
        self,
        image,
        masks
    ):
        """
        image: (3,H,W)
        masks: (25,H,W)
        """

        _, H, W = image.shape

        P = self.patch_size

        # -----------------------------
        # foreground-guided sampling
        # -----------------------------

        if random.random() < self.foreground_prob:

            union_mask = masks.sum(axis=0) > 0

            ys, xs = np.where(union_mask)

            if len(xs) > 0:

                idx = random.randint(
                    0,
                    len(xs) - 1
                )

                cx = xs[idx]
                cy = ys[idx]

            else:
                cx = random.randint(0, W - 1)
                cy = random.randint(0, H - 1)

        else:

            cx = random.randint(0, W - 1)
            cy = random.randint(0, H - 1)

        # -----------------------------
        # compute patch
        # -----------------------------

        x1 = cx - P // 2
        y1 = cy - P // 2

        x1 = max(0, x1)
        y1 = max(0, y1)

        x1 = min(x1, W - P)
        y1 = min(y1, H - P)

        x2 = x1 + P
        y2 = y1 + P

        image_patch = image[
            :,
            y1:y2,
            x1:x2
        ]

        mask_patch = masks[
            :,
            y1:y2,
            x1:x2
        ]

        return image_patch, mask_patch