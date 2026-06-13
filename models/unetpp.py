#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:50:26 2026

@author: navdeepkaushish
"""

import segmentation_models_pytorch as smp


def build_model():

    model = smp.UnetPlusPlus(

        encoder_name="resnet34",

        encoder_weights="imagenet",

        in_channels=3,

        classes=25,

        activation=None

    )

    return model