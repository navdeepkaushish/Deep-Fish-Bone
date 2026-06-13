#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:51:33 2026

@author: navdeepkaushish
"""

import torch

from models.unetpp import build_model


device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)

model = build_model()

model = model.to(device)

x = torch.randn(
    2,
    3,
    512,
    512
).to(device)

with torch.no_grad():

    y = model(x)

print(y.shape)