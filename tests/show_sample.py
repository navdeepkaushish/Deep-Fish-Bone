#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:35:20 2026

@author: navdeepkaushish
"""

import matplotlib.pyplot as plt
import numpy as np


def show_sample(sample):

    image = sample["image"].numpy()
    image = image.transpose(1, 2, 0)

    union_mask = (
        sample["mask"]
        .numpy()
        .sum(axis=0)
    )

    plt.figure(figsize=(10, 10))

    plt.imshow(image)

    plt.imshow(
        union_mask,
        alpha=0.35
    )

    plt.title(
        f"{sample['fish_id']}  "
        f"{sample['genotype']}"
    )

    plt.axis("off")

    plt.show()