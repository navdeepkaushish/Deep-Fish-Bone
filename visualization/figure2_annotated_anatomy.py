#!/usr/bin/env python3

from pathlib import Path
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# 25 segmentation classes
# ============================================================

STRUCTURES = [
    "Br1a", "Br1b", "Br2a", "Br2b",
    "Cb1", "Cb2",
    "Ch1", "Ch2",
    "Cl1", "Cl2",
    "D1", "D2",
    "En1", "En2",
    "Hm1", "Hm2",
    "M1", "M2",
    "N",
    "Oc1", "Oc2",
    "Op1", "Op2",
    "P",
    "VC",
]


# Full names for the figure legend
FULL_NAMES = {
    "Br1a": "Br1a – first branchiostegal ray",
    "Br1b": "Br1b – first branchiostegal ray",
    "Br2a": "Br2a – second branchiostegal ray",
    "Br2b": "Br2b – second branchiostegal ray",
    "Cb1": "Cb1 – ceratobranchial",
    "Cb2": "Cb2 – ceratobranchial",
    "Ch1": "Ch1 – ceratohyal",
    "Ch2": "Ch2 – ceratohyal",
    "Cl1": "Cl1 – cleithrum",
    "Cl2": "Cl2 – cleithrum",
    "D1": "D1 – dentary",
    "D2": "D2 – dentary",
    "En1": "En1 – entopterygoid",
    "En2": "En2 – entopterygoid",
    "Hm1": "Hm1 – hyomandibula",
    "Hm2": "Hm2 – hyomandibula",
    "M1": "M1 – maxilla",
    "M2": "M2 – maxilla",
    "N": "N – notochord",
    "Oc1": "Oc1 – occipital bone",
    "Oc2": "Oc2 – occipital bone",
    "Op1": "Op1 – opercle",
    "Op2": "Op2 – opercle",
    "P": "P – parasphenoid",
    "VC": "VC – vertebral column",
}


# ============================================================
# Fixed 25-class palette
#
# Keep this palette fixed for all manuscript figures.
# Values are RGB in [0, 1].
# ============================================================

COLORS = {
    "Br1a": (0.90, 0.10, 0.10),
    "Br1b": (1.00, 0.45, 0.10),

    "Br2a": (0.85, 0.10, 0.75),
    "Br2b": (0.45, 0.10, 0.85),

    "Cb1": (0.10, 0.55, 0.90),
    "Cb2": (0.10, 0.80, 0.90),

    "Ch1": (0.10, 0.70, 0.35),
    "Ch2": (0.45, 0.80, 0.20),

    "Cl1": (0.90, 0.70, 0.10),
    "Cl2": (1.00, 0.85, 0.30),

    "D1": (0.65, 0.25, 0.15),
    "D2": (0.80, 0.45, 0.25),

    "En1": (0.20, 0.30, 0.80),
    "En2": (0.35, 0.50, 0.95),

    "Hm1": (0.55, 0.15, 0.55),
    "Hm2": (0.75, 0.35, 0.75),

    "M1": (0.10, 0.60, 0.55),
    "M2": (0.30, 0.80, 0.70),

    "N": (0.95, 0.25, 0.50),

    "Oc1": (0.50, 0.50, 0.15),
    "Oc2": (0.70, 0.70, 0.25),

    "Op1": (0.15, 0.35, 0.45),
    "Op2": (0.30, 0.55, 0.65),

    "P": (0.90, 0.35, 0.60),

    "VC": (0.35, 0.35, 0.35),
}


# ============================================================
# IO
# ============================================================

def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path, shape):
    """
    Missing masks are treated as empty masks.
    """
    if not path.exists():
        return np.zeros(shape, dtype=bool)

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return np.zeros(shape, dtype=bool)

    if mask.shape != shape:
        mask = cv2.resize(
            mask,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    return mask > 0


# ============================================================
# Overlay
# ============================================================

def create_overlay(image, masks, alpha=0.72):
    """
    Overlay each structure using the fixed class-specific palette.

    If masks overlap, colors are blended sequentially.
    """
    output = image.astype(np.float32).copy()

    for structure in STRUCTURES:

        mask = masks[structure]

        if not np.any(mask):
            continue

        color = np.array(
            COLORS[structure],
            dtype=np.float32
        ) * 255.0

        output[mask] = (
            (1.0 - alpha) * output[mask]
            + alpha * color
        )

    return np.clip(output, 0, 255).astype(np.uint8)


def add_contours(image, masks, thickness=2):
    """
    Add class-colored contours on top of the filled overlay.
    Improves boundary visibility.
    """
    output = image.copy()

    for structure in STRUCTURES:

        mask = masks[structure]

        if not np.any(mask):
            continue

        rgb = np.array(COLORS[structure]) * 255

        # OpenCV wants BGR
        color = (
            int(rgb[2]),
            int(rgb[1]),
            int(rgb[0]),
        )

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # convert RGB -> BGR temporarily
        tmp = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        cv2.drawContours(
            tmp,
            contours,
            -1,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        output = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    return output


# ============================================================
# Main figure
# ============================================================

def make_figure(image_id, ventral_dir, output_dir, alpha):

    image_folder = ventral_dir / str(image_id)

    if not image_folder.exists():
        raise FileNotFoundError(
            f"\nImage-ID folder not found:\n{image_folder}"
        )

    image_path = image_folder / "image.png"

    if not image_path.exists():
        raise FileNotFoundError(
            f"\nimage.png not found:\n{image_path}"
        )

    print(f"\nImage ID: {image_id}")
    print(f"Folder:   {image_folder}")

    image = load_rgb(image_path)

    h, w = image.shape[:2]

    masks = {}

    present = []
    absent = []

    for structure in STRUCTURES:

        path = image_folder / f"{structure}.png"

        mask = load_mask(
            path,
            (h, w)
        )

        masks[structure] = mask

        if np.any(mask):
            present.append(structure)
        else:
            absent.append(structure)

    print(f"\nAnnotated structures: {len(present)}/25")
    print(", ".join(present))

    if absent:
        print("\nNo reference annotation:")
        print(", ".join(absent))

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        image,
        masks,
        alpha=alpha
    )

    overlay = add_contours(
        overlay,
        masks,
        thickness=2
    )

    # --------------------------------------------------------
    # Layout
    #
    # Top: two image panels
    # Bottom: compact anatomical legend
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(15, 9.5)
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[5.5, 1.8],
        hspace=0.02,
        wspace=0.03,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # --------------------------------------------------------
    # Panel A
    # --------------------------------------------------------

    ax1.imshow(image)
    ax1.axis("off")

    ax1.text(
        0.015,
        0.98,
        "A",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.65,
            edgecolor="none",
            pad=4,
        ),
    )

    ax1.set_title(
        "Brightfield microscopy",
        fontsize=15,
        pad=8,
    )

    # --------------------------------------------------------
    # Panel B
    # --------------------------------------------------------

    ax2.imshow(overlay)
    ax2.axis("off")

    ax2.text(
        0.015,
        0.98,
        "B",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.65,
            edgecolor="none",
            pad=4,
        ),
    )

    ax2.set_title(
        "Expert skeletal annotations",
        fontsize=15,
        pad=8,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    handles = []

    for structure in STRUCTURES:

        # Include only annotated structures in the displayed legend.
        # Comment out this condition if you always want all 25.
        if not np.any(masks[structure]):
            continue

        handles.append(
            Patch(
                facecolor=COLORS[structure],
                edgecolor="black",
                linewidth=0.5,
                label=FULL_NAMES[structure],
            )
        )

    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=5,
        frameon=False,
        fontsize=9.3,
        columnspacing=1.35,
        handlelength=1.4,
        handleheight=1.0,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_png = (
        output_dir /
        f"figure2_annotated_anatomy_{image_id}.png"
    )

    output_pdf = (
        output_dir /
        f"figure2_annotated_anatomy_{image_id}.pdf"
    )

    plt.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    plt.savefig(
        output_pdf,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    plt.close(fig)

    print("\nSaved:")
    print(output_png)
    print(output_pdf)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 2 showing an original zebrafish "
            "brightfield image and the corresponding 25-class "
            "expert skeletal annotation overlay."
        )
    )

    parser.add_argument(
        "--id",
        required=True,
        help="Image ID, e.g. 163424271",
    )

    parser.add_argument(
        "--ventral-dir",
        type=Path,
        default=Path("ventral"),
        help=(
            "Path to ventral dataset directory. "
            "Default: ./ventral"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory. Default: ./figures",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.72,
        help=(
            "Annotation overlay opacity. "
            "Default: 0.72"
        ),
    )

    args = parser.parse_args()

    make_figure(
        image_id=args.id,
        ventral_dir=args.ventral_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()