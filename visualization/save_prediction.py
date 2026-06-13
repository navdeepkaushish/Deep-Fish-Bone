from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_prediction(
    image,
    gt_mask,
    pred_mask,
    save_dir,
    image_id
):

    Path(save_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    image = image.cpu().numpy()
    image = image.transpose(
        1,
        2,
        0
    )

    gt_union = (
        gt_mask.cpu().numpy()
        .sum(axis=0)
        > 0
    )

    pred_union = (
        pred_mask.cpu().numpy()
        .sum(axis=0)
        > 0
    )

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(image)
    ax[1].imshow(
        gt_union,
        alpha=0.4
    )
    ax[1].set_title("GT")
    ax[1].axis("off")

    ax[2].imshow(image)
    ax[2].imshow(
        pred_union,
        alpha=0.4
    )
    ax[2].set_title("Prediction")
    ax[2].axis("off")

    plt.tight_layout()

    plt.savefig(
        Path(save_dir)
        /
        f"{image_id}.png"
    )

    plt.close()