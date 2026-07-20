#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Nested, genotype-stratified GroupKFold splitting at fish level."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class NestedFishSplitter:
    """
    Five outer grouped folds plus one inner grouped validation split.

    Outer test:
        Each fish appears in the outer test set exactly once.

    Inner validation:
        Selected only from the outer-training fish and used for checkpoint
        selection. The outer test set remains untouched until final evaluation.
    """

    def __init__(
        self,
        n_outer_splits=5,
        inner_val_fraction=0.15,
        random_state=42,
    ):
        if n_outer_splits < 2:
            raise ValueError("n_outer_splits must be at least 2")
        if not 0.0 < inner_val_fraction < 0.5:
            raise ValueError("inner_val_fraction must be between 0 and 0.5")

        self.n_outer_splits = n_outer_splits
        self.inner_val_fraction = inner_val_fraction
        self.random_state = random_state

    @staticmethod
    def _check_fish_genotype_consistency(dataframe):
        n_genotypes = dataframe.groupby("fish_id")["genotype"].nunique()
        bad = n_genotypes[n_genotypes > 1]
        if not bad.empty:
            raise ValueError(
                "Each fish_id must map to exactly one genotype. "
                f"Inconsistent fish IDs: {bad.index.tolist()}"
            )

    @staticmethod
    def _assert_disjoint(dataframe, train_idx, val_idx, test_idx, fold):
        train_fish = set(dataframe.iloc[train_idx]["fish_id"])
        val_fish = set(dataframe.iloc[val_idx]["fish_id"])
        test_fish = set(dataframe.iloc[test_idx]["fish_id"])

        if not train_fish.isdisjoint(val_fish):
            raise RuntimeError(f"Fish leakage between train and val in fold {fold}")
        if not train_fish.isdisjoint(test_fish):
            raise RuntimeError(f"Fish leakage between train and test in fold {fold}")
        if not val_fish.isdisjoint(test_fish):
            raise RuntimeError(f"Fish leakage between val and test in fold {fold}")

    def _inner_split(self, dataframe, outer_train_idx, fold):
        outer_train_df = dataframe.iloc[outer_train_idx]

        fish_counts = (
            outer_train_df[["fish_id", "genotype"]]
            .drop_duplicates()
            .groupby("genotype")["fish_id"]
            .nunique()
        )
        max_splits = int(fish_counts.min())
        if max_splits < 2:
            raise ValueError(
                "At least one genotype has fewer than two fish in the "
                f"outer-training data of fold {fold}: {fish_counts.to_dict()}"
            )

        requested_splits = max(2, int(round(1.0 / self.inner_val_fraction)))
        n_inner_splits = min(requested_splits, max_splits)

        inner = StratifiedGroupKFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=self.random_state + fold + 1,
        )

        y = outer_train_df["genotype"].to_numpy()
        groups = outer_train_df["fish_id"].to_numpy()
        dummy_x = np.zeros(len(outer_train_df))

        candidates = []
        for inner_train_pos, inner_val_pos in inner.split(dummy_x, y, groups):
            fraction = len(inner_val_pos) / len(dataframe)
            distance = abs(fraction - self.inner_val_fraction)
            candidates.append((distance, inner_train_pos, inner_val_pos))

        _, inner_train_pos, inner_val_pos = min(
            candidates, key=lambda item: item[0]
        )

        train_idx = np.asarray(outer_train_idx)[inner_train_pos]
        val_idx = np.asarray(outer_train_idx)[inner_val_pos]
        return train_idx, val_idx

    def split(self, dataframe):
        self._check_fish_genotype_consistency(dataframe)

        fish_counts = (
            dataframe[["fish_id", "genotype"]]
            .drop_duplicates()
            .groupby("genotype")["fish_id"]
            .nunique()
        )
        if fish_counts.min() < self.n_outer_splits:
            raise ValueError(
                "Every genotype needs at least n_outer_splits unique fish. "
                f"Fish counts: {fish_counts.to_dict()}"
            )

        outer = StratifiedGroupKFold(
            n_splits=self.n_outer_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        y = dataframe["genotype"].to_numpy()
        groups = dataframe["fish_id"].to_numpy()
        dummy_x = np.zeros(len(dataframe))

        for fold, (outer_train_idx, test_idx) in enumerate(
            outer.split(dummy_x, y, groups)
        ):
            train_idx, val_idx = self._inner_split(
                dataframe=dataframe,
                outer_train_idx=outer_train_idx,
                fold=fold,
            )

            self._assert_disjoint(
                dataframe=dataframe,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                fold=fold,
            )

            yield fold, train_idx, val_idx, np.asarray(test_idx)
