#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:38:14 2026

@author: navdeepkaushish
"""

from pathlib import Path
import pandas as pd


class Metadata:

    def __init__(self, csv_path):

        csv_path = Path(csv_path)

        self.df = pd.read_csv(csv_path)

        required = [
            "image_id",
            "fish_id",
            "genotype",
            "quality"
        ]

        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")

    def dataframe(self):
        return self.df.copy()

    def __len__(self):
        return len(self.df)