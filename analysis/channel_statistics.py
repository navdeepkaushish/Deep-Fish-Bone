#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:19:21 2026

@author: navdeepkaushish
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configs.structure_names import STRUCTURES


ROOT_DIR = "/Users/navdeepkaushish/Documents/missing_bone_structures/deep-fish-bone/ventral"


def main():

    image_dirs = sorted(
        [
            p
            for p in Path(ROOT_DIR).iterdir()
            if p.is_dir()
        ]
    )

    print(
        f"Found {len(image_dirs)} images"
    )

    stats = {}

    for structure in STRUCTURES:

        stats[structure] = {

            "presence_count": 0,

            "positive_pixels": 0,

            "total_pixels": 0

        }

    # -------------------------
    # Iterate images
    # -------------------------

    for image_dir in image_dirs:

        for structure in STRUCTURES:

            mask_path = (
                image_dir
                /
                f"{structure}.png"
            )

            if not mask_path.exists():

                raise FileNotFoundError(
                    mask_path
                )

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            mask = (
                mask > 0
            ).astype(
                np.uint8
            )

            pos_pixels = mask.sum()

            stats[structure][
                "positive_pixels"
            ] += pos_pixels

            stats[structure][
                "total_pixels"
            ] += mask.size

            if pos_pixels > 0:

                stats[structure][
                    "presence_count"
                ] += 1

    # -------------------------
    # Create dataframe
    # -------------------------

    rows = []

    n_images = len(
        image_dirs
    )

    for structure in STRUCTURES:

        presence = stats[
            structure
        ][
            "presence_count"
        ]

        positive_pixels = stats[
            structure
        ][
            "positive_pixels"
        ]

        total_pixels = stats[
            structure
        ][
            "total_pixels"
        ]

        pixel_fraction = (

            positive_pixels

            /

            total_pixels

        )

        rows.append(

            {

                "structure":
                structure,

                "presence_count":
                presence,

                "presence_fraction":
                round(
                    presence
                    /
                    n_images,
                    4
                ),

                "positive_pixels":
                int(
                    positive_pixels
                ),

                "pixel_fraction":
                pixel_fraction

            }

        )

    df = pd.DataFrame(
        rows
    )

    df = df.sort_values(

        by="pixel_fraction",

        ascending=False

    )

    print()

    print(df)

    Path(
        "outputs"
    ).mkdir(
        exist_ok=True
    )

    df.to_csv(

        "outputs/channel_statistics.csv",

        index=False

    )

    print(
        "\nSaved:"
    )

    print(
        "outputs/channel_statistics.csv"
    )
    


if __name__ == "__main__":

    main()