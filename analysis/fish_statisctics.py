#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:02:48 2026

@author: navdeepkaushish
"""

import pandas as pd


def main():

    df = pd.read_csv(
        "metadata/metadata.csv"
    )

    print()

    print(
        "Total Images:",
        len(df)
    )

    print()

    print(
        "Unique Fish:",
        df["fish_id"].nunique()
    )

    print()

    print(
        "Genotype Counts"
    )

    print(
        df.groupby(
            "genotype"
        )["fish_id"]
        .nunique()
    )

    print()

    images_per_fish = (

        df.groupby(
            "fish_id"
        )

        .size()

    )

    print()

    print(
        "Images Per Fish"
    )

    print(
        images_per_fish.describe()
    )

    print()

    print(
        "Quality Distribution"
    )

    print(
        df["quality"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":

    main()