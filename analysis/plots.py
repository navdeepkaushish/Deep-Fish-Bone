#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:22:15 2026

@author: navdeepkaushish
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputs/channel_statistics.csv")
#========= plots ==========================================================    
plt.figure(
    figsize=(12,6)
)

plt.bar(

    df["structure"],

    df["presence_count"]

)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    "outputs/presence_count.png"
)

plt.close()


plt.figure(
    figsize=(12,6)
)

plt.bar(

    df["structure"],

    df["pixel_fraction"]

)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    "outputs/pixel_fraction.png"
)

plt.close()