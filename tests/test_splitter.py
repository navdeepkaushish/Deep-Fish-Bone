#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:39:58 2026

@author: navdeepkaushish
"""

from datasets.metadata import Metadata
from datasets.splitter import FishSplitter


meta = Metadata(
    "metadata/metadata.csv"
)

df = meta.dataframe()

splitter = FishSplitter(
    n_splits=5
)

for fold, train_idx, val_idx in splitter.split(df):

    train = df.iloc[train_idx]
    val = df.iloc[val_idx]

    print("=" * 60)
    print(f"Fold {fold}")

    print()

    print(
        "Train fish:",
        train["fish_id"].nunique()
    )

    print(
        "Val fish:",
        val["fish_id"].nunique()
    )

    overlap = set(
        train["fish_id"]
    ) & set(
        val["fish_id"]
    )

    print()

    print(
        "Fish overlap:",
        len(overlap)
    )

    print()

    print(
        train["genotype"]
        .value_counts()
    )

    print()

    print(
        val["genotype"]
        .value_counts()
    )