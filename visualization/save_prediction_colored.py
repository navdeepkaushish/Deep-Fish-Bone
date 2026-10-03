from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from configs.structure_names import STRUCTURES


# Fixed color palette.
# Same color is used for each structure in GT and Prediction.
COLORS = {
    "Br1a": "#1f77b4",
    "Br1b": "#6baed6",
    "Br2a": "#d62728",
    "Br2b": "#ff9896",

    "Cb1": "#2ca02c",
    "Cb2": "#98df8a",

    "Ch1": "#9467bd",
    "Ch2": "#c5b0d5",

    "Cl1": "#8c564b",
    "Cl2": "#c49c94",

    "D1": "#e377c2",
    "D2": "#f7b6d2",

    "En1": "#7f7f7f",
    "En2": "#c7c7c7",

    "Hm1": "#bcbd22",
    "Hm2": "#dbdb8d",

    "M1": "#17becf",
    "M2": "#9edae5",

    "N": "#ff7f0e",

    "Oc1": "#393b79",
    "Oc2": "#637939",

    "Op1": "#8c6d31",
    "Op2": "#843c39",

    "P": "#7b4173",

    "VC": "#000000",
}


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array(
        [
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        ],
        dtype=np.float32,
    ) / 255.0


def make_colored_overlay(
    image,
    masks,
    alpha=0.55,
):
    """
    Overlay each anatomical structure using a fixed class-specific color.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), values expected in [0, 1].
    masks : np.ndarray
        Binary masks, shape (C, H, W).
    alpha : float
        Overlay opacity.

    Returns
    -------
    overlay : np.ndarray
        RGB overlay image.
    """
    overlay = image.copy().astype(np.float32)

    for class_idx, structure in enumerate(STRUCTURES):
        mask = masks[class_idx] > 0.5

        if not np.any(mask):
            continue

        color = _hex_to_rgb(COLORS[structure])

        overlay[mask] = (
            (1.0 - alpha) * overlay[mask]
            + alpha * color
        )

    return np.clip(overlay, 0.0, 1.0)


def save_prediction_colored(
    image,
    gt_mask,
    pred_mask,
    save_dir,
    image_id,
    alpha=0.55,
    show_legend=True,
):
    """
    Save microscopy image, colored GT overlay, and colored prediction overlay.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Convert tensors to numpy
    # ---------------------------------------------------------
    image_np = (
        image.detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    gt_np = (
        gt_mask.detach()
        .cpu()
        .numpy()
    )

    pred_np = (
        pred_mask.detach()
        .cpu()
        .numpy()
    )

    # Ensure image is displayable
    image_np = np.clip(image_np, 0.0, 1.0)

    gt_overlay = make_colored_overlay(
        image_np,
        gt_np,
        alpha=alpha,
    )

    pred_overlay = make_colored_overlay(
        image_np,
        pred_np,
        alpha=alpha,
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    ax[0].imshow(image_np)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(gt_overlay)
    ax[1].set_title("Ground truth")
    ax[1].axis("off")

    ax[2].imshow(pred_overlay)
    ax[2].set_title("Prediction")
    ax[2].axis("off")

    # ---------------------------------------------------------
    # Common legend
    # ---------------------------------------------------------
    if show_legend:
        legend_handles = [
            Patch(
                facecolor=COLORS[s],
                edgecolor="none",
                label=s,
            )
            for s in STRUCTURES
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=9,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

        plt.tight_layout(
            rect=[0, 0.10, 1, 1]
        )
    else:
        plt.tight_layout()

    output_path = (
        save_dir
        / f"{image_id}_colored.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path