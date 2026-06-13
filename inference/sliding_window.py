#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 10:04:05 2026

@author: navdeepkaushish
"""

import torch


def sliding_window_predict(
    model,
    image,
    device,
    patch_size=512,
    stride=256,
    num_classes=25
):
    """
    image: torch.Tensor, shape (3, H, W)
    returns: torch.Tensor, shape (num_classes, H, W), probabilities
    """

    model.eval()

    _, H, W = image.shape

    output = torch.zeros(
        (num_classes, H, W),
        dtype=torch.float32,
        device=device
    )

    count_map = torch.zeros(
        (1, H, W),
        dtype=torch.float32,
        device=device
    )

    image = image.to(device)

    y_positions = list(range(0, H - patch_size + 1, stride))
    x_positions = list(range(0, W - patch_size + 1, stride))

    if y_positions[-1] != H - patch_size:
        y_positions.append(H - patch_size)

    if x_positions[-1] != W - patch_size:
        x_positions.append(W - patch_size)

    with torch.no_grad():

        for y in y_positions:
            for x in x_positions:

                patch = image[
                    :,
                    y:y + patch_size,
                    x:x + patch_size
                ]

                patch = patch.unsqueeze(0)

                logits = model(patch)

                probs = torch.sigmoid(logits)[0]

                output[
                    :,
                    y:y + patch_size,
                    x:x + patch_size
                ] += probs

                count_map[
                    :,
                    y:y + patch_size,
                    x:x + patch_size
                ] += 1

    output = output / count_map

    return output