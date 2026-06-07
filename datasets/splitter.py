#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:39:12 2026

@author: navdeepkaushish
"""

from sklearn.model_selection import StratifiedGroupKFold


class FishSplitter:

    def __init__(
        self,
        n_splits=5,
        random_state=42
    ):
        self.sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    def split(self, dataframe):

        X = dataframe.index.values

        y = dataframe["genotype"].values

        groups = dataframe["fish_id"].values

        for fold, (train_idx, val_idx) in enumerate(

            self.sgkf.split(
                X,
                y,
                groups
            )

        ):

            yield fold, train_idx, val_idx