#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skimage.measure import (
    regionprops,
    label
)

from configs.structure_names import (
    STRUCTURES
)


ROOT_DIR = "ventral"
METADATA = "metadata/metadata.csv"


def extract_features(mask):

    features = {

        "area": 0,
        "perimeter": 0,

        "centroid_x": np.nan,
        "centroid_y": np.nan,

        "bbox_width": 0,
        "bbox_height": 0,

        "major_axis": 0,
        "minor_axis": 0,

        "eccentricity": np.nan,

        "orientation": np.nan,

        "solidity": np.nan

    }

    if mask.sum() == 0:

        return features

    labeled = label(mask)

    props = regionprops(
        labeled
    )

    if len(props) == 0:

        return features

    region = max(
        props,
        key=lambda x: x.area
    )

    minr, minc, maxr, maxc = (
        region.bbox
    )

    features["area"] = (
        region.area
    )

    features["perimeter"] = (
        region.perimeter
    )

    features["centroid_x"] = (
        region.centroid[1]
    )

    features["centroid_y"] = (
        region.centroid[0]
    )

    features["bbox_width"] = (
        maxc - minc
    )

    features["bbox_height"] = (
        maxr - minr
    )

    features["major_axis"] = (
        region.major_axis_length
    )

    features["minor_axis"] = (
        region.minor_axis_length
    )

    features["eccentricity"] = (
        region.eccentricity
    )

    features["orientation"] = (
        region.orientation
    )

    features["solidity"] = (
        region.solidity
    )

    return features


def main():

    metadata = pd.read_csv(
        METADATA
    )

    rows = []

    for _, sample in metadata.iterrows():

        image_id = str(
            sample["image_id"]
        )

        fish_id = (
            sample["fish_id"]
        )

        genotype = (
            sample["genotype"]
        )

        image_dir = (
            Path(ROOT_DIR)
            /
            image_id
        )

        for structure in STRUCTURES:

            mask_file = (
                image_dir
                /
                f"{structure}.png"
            )

            if not mask_file.exists():

                continue

            mask = cv2.imread(

                str(mask_file),

                cv2.IMREAD_GRAYSCALE

            )

            mask = (
                mask > 0
            ).astype(
                np.uint8
            )

            features = (
                extract_features(
                    mask
                )
            )

            row = {

                "image_id":
                image_id,

                "fish_id":
                fish_id,

                "genotype":
                genotype,

                "structure":
                structure

            }

            row.update(
                features
            )

            rows.append(
                row
            )

    df = pd.DataFrame(
        rows
    )

    Path(
        "outputs"
    ).mkdir(
        exist_ok=True
    )

    output_file = (
        "outputs/"
        "morphometry.csv"
    )

    df.to_csv(

        output_file,

        index=False

    )

    print()

    print(
        f"Saved:"
    )

    print(
        output_file
    )

    print()

    print(
        df.head()
    )


if __name__ == "__main__":

    main()