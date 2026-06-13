#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:05:25 2026

@author: navdeepkaushish
"""

import json
import numpy as np
import cv2
from PIL import Image, ImageDraw

def labelme_json_to_binary_mask(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get image dimensions
    img_h = data['imageHeight']
    img_w = data['imageWidth']

    # Create blank binary mask
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], color=255)

        elif shape['shape_type'] == 'rectangle':
            (x1, y1), (x2, y2) = shape['points']
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

    cv2.imwrite(output_path, mask)
    print(f"Saved binary mask to {output_path} — shape: {mask.shape}")

# Usage
labelme_json_to_binary_mask("ventral/525419017/VC.json", "binary_mask.png")