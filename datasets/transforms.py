#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:08 2026

@author: navdeepkaushish
"""

import albumentations as A
import cv2


def get_train_transforms():

    return A.Compose([

    A.HorizontalFlip(
        p=0.5
    ),

    A.Rotate(
        limit=12,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.10,
        contrast_limit=0.10,
        p=0.4
    ),

    A.GaussianBlur(
        blur_limit=(3,5),
        p=0.25
    )

])


def get_valid_transforms():

    return A.Compose([])