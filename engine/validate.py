#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:02:09 2026

@author: navdeepkaushish
"""

from tqdm import tqdm

import numpy as np
import torch

from metrics.dice import (
    multiclass_dice
)

from metrics.per_structure_dice import (
    per_structure_dice
)

from visualization.save_prediction import (
    save_prediction
)


def validate(
    model,
    loader,
    criterion,
    device,
    epoch=None
):

    model.eval()

    running_loss = 0
    running_dice = 0

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

            images = batch["image"].to(
                device
            )

            masks = batch["mask"].to(
                device
            )

            logits = model(images)

            loss = criterion(
                logits,
                masks
            )

            mean_dice = multiclass_dice(
                logits,
                masks
            )

            running_loss += (
                loss.item()
            )

            running_dice += (
                mean_dice
            )

            (
                batch_dice_sum,
                batch_counts
            ) = per_structure_dice(

                logits,

                masks

            )

            class_dice_sum += np.array(
                batch_dice_sum
            )

            class_valid_count += np.array(
                batch_counts
            )

            # ----------------------------------
            # Save prediction visualization
            # ----------------------------------

            if (
                batch_idx == 0
                and epoch is not None
                and (
                    epoch == 0
                    or (epoch + 1) % 5 == 0
                )
            ):

                probs = torch.sigmoid(
                    logits
                )

                preds = (
                    probs > 0.5
                ).float()

                save_prediction(

                    images[0].cpu(),

                    masks[0].cpu(),

                    preds[0].cpu(),

                    save_dir=
                    f"outputs/predictions/epoch_{epoch+1:03d}",

                    image_id=
                    str(
                        batch["image_id"][0]
                    )

                )

    avg_loss = (
        running_loss
        /
        len(loader)
    )

    avg_dice = (
        running_dice
        /
        len(loader)
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

    return (

        avg_loss,

        avg_dice,

        avg_class_dice,

        class_valid_count.tolist()

    )