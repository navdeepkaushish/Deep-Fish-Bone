#!/usr/bin/env python3

import pandas as pd


def main():

    df = pd.read_csv(
        "metadata/metadata.csv"
    )

    print("\n")

    print(
        "Total Images:",
        len(df)
    )

    print(
        "Unique Fish:",
        df["fish_id"].nunique()
    )

    print("\nGenotypes")

    print(

        df.groupby(
            "genotype"
        )["fish_id"]

        .nunique()

    )

    print("\nImages Per Fish")

    images_per_fish = (

        df.groupby(
            "fish_id"
        )

        .size()

    )

    print(
        images_per_fish.describe()
    )

    print("\nQuality")

    print(

        df["quality"]

        .value_counts()

        .sort_index()

    )


if __name__ == "__main__":

    main()