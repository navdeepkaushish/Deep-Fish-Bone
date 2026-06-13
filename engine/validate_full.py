#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 10:32:27 2026

@author: navdeepkaushish
"""

from tqdm import tqdm

import numpy as np
import torch

from inference.sliding_window import (
    sliding_window_predict
)

from metrics.per_structure_dice import (
    per_structure_dice
)

from visualization.save_prediction import (
    save_prediction
)


def validate_full_image(
    model,
    loader,
    device,
    epoch=None,
    patch_size=512,
    stride=256,
    threshold=0.5
):

    model.eval()

    n_classes = 25

    class_dice_sum = np.zeros(
        n_classes
    )

    class_valid_count = np.zeros(
        n_classes
    )

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            tqdm(loader)
        ):

            image = batch["image"][0]

            mask = batch["mask"][0].to(
                device
            )

            probs = sliding_window_predict(

                model=model,

                image=image,

                device=device,

                patch_size=patch_size,

                stride=stride

            )

            logits = torch.logit(

                probs.clamp(
                    1e-6,
                    1 - 1e-6
                )

            ).unsqueeze(0)

            mask_batch = (
                mask.unsqueeze(0)
            )

            (
                dice_sum,
                counts
            ) = per_structure_dice(

                logits,

                mask_batch,

                threshold=threshold

            )

            class_dice_sum += np.array(
                dice_sum
            )

            class_valid_count += np.array(
                counts
            )

            # --------------------------
            # Save prediction
            # --------------------------

            if (
                batch_idx == 0
                and epoch is not None
            ):

                preds = (
                    probs > threshold
                ).float()

                save_prediction(

                    image.cpu(),

                    mask.cpu(),

                    preds.cpu(),

                    save_dir=
                    f"outputs/full_predictions/epoch_{epoch+1:03d}",

                    image_id=
                    str(
                        batch["image_id"][0]
                    )

                )

    avg_class_dice = []

    for dice_sum, count in zip(

        class_dice_sum,

        class_valid_count

    ):

        if count == 0:

            avg_class_dice.append(
                None
            )

        else:

            avg_class_dice.append(

                float(
                    dice_sum
                    /
                    count
                )

            )

    valid_scores = [

        score

        for score in avg_class_dice

        if score is not None

    ]

    mean_dice = float(
        np.mean(
            valid_scores
        )
    )

    return (

        mean_dice,

        avg_class_dice,

        class_valid_count.tolist()

    )