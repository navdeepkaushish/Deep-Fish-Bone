#!/usr/bin/env python3
"""
Create nested grouped cross-validation splits from metadata.csv.

Expected metadata columns
-------------------------
image_id, fish_id, genotype, quality

Design
------
Outer loop:
    StratifiedGroupKFold with fish_id as the grouping variable.
    The outer test fold is never used for training, checkpoint selection,
    threshold tuning, or early stopping.

Inner split:
    One stratified grouped holdout is selected from the outer-training data.
    It is used for model selection and early stopping.

Outputs
-------
nested_cv_assignments.csv
    One row per image per outer fold, with split = train / val / test.

nested_cv_summary.csv
    Fish- and image-level counts by outer fold, split, and genotype.

nested_cv_fish_assignments.csv
    One row per fish per outer fold.

The default 5 outer folds ensure that each fish appears in the outer test
set once. The default inner validation fraction is 0.15 of the complete
dataset approximately; because splits are made by fish, exact proportions
may vary slightly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_COLUMNS = {"image_id", "fish_id", "genotype"}


def validate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["fish_id"] = df["fish_id"].astype(str)
    df["genotype"] = df["genotype"].astype(str).str.lower().str.strip()

    if df["image_id"].duplicated().any():
        duplicates = df.loc[df["image_id"].duplicated(), "image_id"].tolist()[:10]
        raise ValueError(f"Duplicate image_id values detected, e.g. {duplicates}")

    # Every fish must have one and only one genotype.
    genotype_counts = df.groupby("fish_id")["genotype"].nunique()
    inconsistent = genotype_counts[genotype_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Some fish IDs have multiple genotype labels: "
            f"{inconsistent.index.tolist()}"
        )

    return df


def choose_inner_split(
    outer_train_df: pd.DataFrame,
    inner_val_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Choose one grouped, genotype-stratified inner validation split.

    StratifiedGroupKFold is used with n_splits approximately equal to
    1 / inner_val_fraction. Among its candidate folds, the split whose
    validation image fraction is closest to the requested fraction is selected.
    """
    if not 0 < inner_val_fraction < 0.5:
        raise ValueError("inner_val_fraction must be between 0 and 0.5")

    n_splits = max(2, int(round(1.0 / inner_val_fraction)))

    fish_per_genotype = (
        outer_train_df[["fish_id", "genotype"]]
        .drop_duplicates()
        .groupby("genotype")["fish_id"]
        .nunique()
    )
    max_valid_splits = int(fish_per_genotype.min())

    if max_valid_splits < 2:
        raise ValueError(
            "The outer-training subset contains fewer than two fish in at "
            "least one genotype, so a stratified grouped inner split is impossible."
        )

    n_splits = min(n_splits, max_valid_splits)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    candidates = []
    y = outer_train_df["genotype"].to_numpy()
    groups = outer_train_df["fish_id"].to_numpy()

    for train_pos, val_pos in splitter.split(
        X=np.zeros(len(outer_train_df)),
        y=y,
        groups=groups,
    ):
        val_fraction = len(val_pos) / len(outer_train_df)
        distance = abs(val_fraction - inner_val_fraction)
        candidates.append((distance, train_pos, val_pos))

    _, train_pos, val_pos = min(candidates, key=lambda item: item[0])
    return train_pos, val_pos


def assert_no_leakage(fold_df: pd.DataFrame, outer_fold: int) -> None:
    fish_sets = {
        split: set(fold_df.loc[fold_df["split"] == split, "fish_id"])
        for split in ("train", "val", "test")
    }

    assert fish_sets["train"].isdisjoint(fish_sets["val"]), (
        f"Fish leakage between train and val in outer fold {outer_fold}"
    )
    assert fish_sets["train"].isdisjoint(fish_sets["test"]), (
        f"Fish leakage between train and test in outer fold {outer_fold}"
    )
    assert fish_sets["val"].isdisjoint(fish_sets["test"]), (
        f"Fish leakage between val and test in outer fold {outer_fold}"
    )

    assigned = (
        fold_df.groupby("image_id")["split"]
        .nunique()
    )
    assert (assigned == 1).all(), (
        f"Some images have multiple split assignments in outer fold {outer_fold}"
    )


def build_nested_splits(
    df: pd.DataFrame,
    outer_splits: int,
    inner_val_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    fish_per_genotype = (
        df[["fish_id", "genotype"]]
        .drop_duplicates()
        .groupby("genotype")["fish_id"]
        .nunique()
    )

    if fish_per_genotype.min() < outer_splits:
        raise ValueError(
            "Each genotype needs at least outer_splits unique fish. "
            f"Current fish counts: {fish_per_genotype.to_dict()}"
        )

    outer = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    all_folds = []
    y = df["genotype"].to_numpy()
    groups = df["fish_id"].to_numpy()

    for outer_fold, (outer_train_pos, test_pos) in enumerate(
        outer.split(X=np.zeros(len(df)), y=y, groups=groups),
        start=1,
    ):
        outer_train_df = df.iloc[outer_train_pos].copy()
        test_df = df.iloc[test_pos].copy()

        inner_train_pos, val_pos = choose_inner_split(
            outer_train_df=outer_train_df,
            inner_val_fraction=inner_val_fraction,
            random_state=random_state + outer_fold,
        )

        train_df = outer_train_df.iloc[inner_train_pos].copy()
        val_df = outer_train_df.iloc[val_pos].copy()

        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        fold_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        fold_df["outer_fold"] = outer_fold

        assert_no_leakage(fold_df, outer_fold)
        all_folds.append(fold_df)

    assignments = pd.concat(all_folds, ignore_index=True)
    column_order = [
        "outer_fold", "split", "image_id", "fish_id", "genotype"
    ]
    remaining = [c for c in assignments.columns if c not in column_order]
    return assignments[column_order + remaining]


def make_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    image_counts = (
        assignments
        .groupby(["outer_fold", "split", "genotype"])
        .size()
        .rename("n_images")
        .reset_index()
    )

    fish_counts = (
        assignments[["outer_fold", "split", "fish_id", "genotype"]]
        .drop_duplicates()
        .groupby(["outer_fold", "split", "genotype"])
        .size()
        .rename("n_fish")
        .reset_index()
    )

    return fish_counts.merge(
        image_counts,
        on=["outer_fold", "split", "genotype"],
        how="outer",
    ).sort_values(["outer_fold", "split", "genotype"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nested_cv_splits"),
    )
    parser.add_argument(
        "--outer-splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--inner-val-fraction",
        type=float,
        default=0.15,
        help=(
            "Target validation fraction relative to the full dataset. "
            "Exact values may vary because splitting is performed by fish."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    df = validate_metadata(pd.read_csv(args.metadata))
    assignments = build_nested_splits(
        df=df,
        outer_splits=args.outer_splits,
        inner_val_fraction=args.inner_val_fraction,
        random_state=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    assignment_path = args.output_dir / "nested_cv_assignments.csv"
    fish_path = args.output_dir / "nested_cv_fish_assignments.csv"
    summary_path = args.output_dir / "nested_cv_summary.csv"
    config_path = args.output_dir / "nested_cv_config.json"

    assignments.to_csv(assignment_path, index=False)

    fish_assignments = (
        assignments[
            ["outer_fold", "split", "fish_id", "genotype"]
        ]
        .drop_duplicates()
        .sort_values(["outer_fold", "split", "genotype", "fish_id"])
    )
    fish_assignments.to_csv(fish_path, index=False)

    summary = make_summary(assignments)
    summary.to_csv(summary_path, index=False)

    config = {
        "metadata": str(args.metadata),
        "outer_splits": args.outer_splits,
        "inner_val_fraction_target": args.inner_val_fraction,
        "seed": args.seed,
        "n_images": int(df["image_id"].nunique()),
        "n_fish": int(df["fish_id"].nunique()),
        "genotype_fish_counts": (
            df[["fish_id", "genotype"]]
            .drop_duplicates()
            .groupby("genotype")["fish_id"]
            .nunique()
            .to_dict()
        ),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nNested CV assignments created successfully.\n")
    print(summary.to_string(index=False))
    print(f"\nAssignments: {assignment_path}")
    print(f"Fish assignments: {fish_path}")
    print(f"Summary: {summary_path}")
    print(f"Configuration: {config_path}")


if __name__ == "__main__":
    main()
