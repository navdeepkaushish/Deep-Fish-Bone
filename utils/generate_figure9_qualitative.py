# -*- coding: utf-8 -*-

"""
Generate Figure 9: qualitative comparison of final segmentation models.

Layout:
    Image | Ground truth | Baseline | Anatomy-aware | Class-specific Br2

Workflow:
1. Run a lightweight pass over all 190 independent-test records.
   Only scalar Dice values are stored.
2. Select four representative cases objectively.
3. Re-run inference only for those four selected images.
4. Create colored 25-class overlays.
5. Save PNG, PDF, and CSV describing selected cases.

Existing experiment outputs are never modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch


# ============================================================
# Repository setup
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from configs.structure_names import STRUCTURES
from engine.dataloaders import build_full_image_loader
from inference.sliding_window import sliding_window_predict
from models.unetpp import build_model


# ============================================================
# Experiment locations
# ============================================================

EXPERIMENTS = {
    "baseline": {
        "label": "Baseline",
        "root": ROOT / "outputs" / "baseline_dice_focal",
    },

    "anatomy": {
        "label": "Anatomy-aware",
        "root": ROOT / "outputs" / "anatomy_aware_dice_focal",
    },

    "class_specific": {
        "label": "Class-specific Br2",
        "root": ROOT / "outputs" / "class_specific_br2",
    },
}


OUTPUT_DIR = ROOT / "outputs" / "figure9"

BR2_INDICES = [
    STRUCTURES.index("Br2a"),
    STRUCTURES.index("Br2b"),
]


# ============================================================
# Fixed colors for all structures
# ============================================================

COLORS = {
    "Br1a": "#1f77b4",
    "Br1b": "#6baed6",

    "Br2a": "#d62728",
    "Br2b": "#ff7f0e",

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

    "N": "#ffbb78",

    "Oc1": "#393b79",
    "Oc2": "#637939",

    "Op1": "#8c6d31",
    "Op2": "#843c39",

    "P": "#7b4173",

    "VC": "#111111",
}


# ============================================================
# Small helpers
# ============================================================

def norm(value):
    """
    Normalize image IDs.
    Example: 163430839.0 -> 163430839
    """
    s = str(value).strip()

    if s.endswith(".0"):
        s = s[:-2]

    return s


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")

    return np.array(
        [
            int(h[0:2], 16),
            int(h[2:4], 16),
            int(h[4:6], 16),
        ],
        dtype=np.float32,
    ) / 255.0


def normalize_image(image_tensor):
    """
    Convert CxHxW torch tensor to HxWx3 float image.
    """

    image = (
        image_tensor.detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
        .astype(np.float32)
    )

    min_val = float(image.min())
    max_val = float(image.max())

    if min_val < 0.0 or max_val > 1.0:
        if max_val > min_val:
            image = (
                image - min_val
            ) / (
                max_val - min_val
            )

    return np.clip(
        image,
        0.0,
        1.0,
    )


# ============================================================
# Colored overlays
# ============================================================

def make_colored_overlay(
    image,
    masks,
    alpha=0.75,
):
    """
    Overlay each of the 25 structures with its own fixed color.

    image:
        H x W x 3

    masks:
        C x H x W
    """

    overlay = image.copy().astype(
        np.float32
    )

    for class_idx, structure in enumerate(
        STRUCTURES
    ):

        mask = masks[class_idx] > 0.5

        if not np.any(mask):
            continue

        color = hex_to_rgb(
            COLORS[structure]
        )

        overlay[mask] = (
            (1.0 - alpha) * overlay[mask]
            + alpha * color
        )

    return np.clip(
        overlay,
        0.0,
        1.0,
    )


# ============================================================
# Frozen fold reconstruction
# ============================================================

def exact_test_indices(
    metadata_csv,
    manifest_path,
):
    """
    Recover test indices from the frozen split manifest.
    """

    metadata = pd.read_csv(
        metadata_csv
    )

    manifest = pd.read_csv(
        manifest_path
    )

    metadata_ids = (
        metadata["image_id"]
        .map(norm)
    )

    lookup = {
        image_id: idx
        for idx, image_id
        in enumerate(metadata_ids)
    }

    test_ids = (
        manifest.loc[
            manifest["split"] == "test",
            "image_id",
        ]
        .map(norm)
        .tolist()
    )

    missing = [
        image_id
        for image_id in test_ids
        if image_id not in lookup
    ]

    if missing:
        raise RuntimeError(
            f"Missing test IDs in metadata: {missing[:10]}"
        )

    return [
        lookup[image_id]
        for image_id in test_ids
    ]


# ============================================================
# Test loader
# ============================================================

def build_test_loader(fold):
    """
    Build exactly the same full-image test loader used during final evaluation.
    """

    fold_dir = (
        EXPERIMENTS["baseline"]["root"]
        / f"fold_{fold}"
    )

    config_path = (
        fold_dir
        / "config_used.json"
    )

    cfg = load_json(
        config_path
    )

    metadata_csv = (
        ROOT
        / cfg["metadata_csv"]
    )

    manifest_path = (
        ROOT
        / cfg["manifest_root"]
        / f"fold_{fold}"
        / "split_manifest.csv"
    )

    test_indices = (
        exact_test_indices(
            metadata_csv,
            manifest_path,
        )
    )

    loader = build_full_image_loader(
        root_dir=str(
            ROOT / cfg["root_dir"]
        ),
        metadata_csv=str(
            metadata_csv
        ),
        indices=test_indices,
        resize_to=tuple(
            cfg["resize_to"]
        ),
        num_workers=0,
    )

    return loader, cfg


# ============================================================
# Model loading
# ============================================================

def load_model(
    experiment,
    fold,
    device,
):
    checkpoint_path = (
        EXPERIMENTS[experiment]["root"]
        / f"fold_{fold}"
        / "checkpoints"
        / "best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = build_model().to(
        device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model


# ============================================================
# Dice functions
# ============================================================

def binary_dice(
    prediction,
    ground_truth,
    eps=1e-6,
):
    """
    Dice only when GT structure is present.
    """

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    if not ground_truth.any():
        return None

    intersection = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    return float(
        (
            2.0 * intersection
            + eps
        )
        /
        (
            prediction.sum()
            + ground_truth.sum()
            + eps
        )
    )


def mean_present_dice(
    prediction,
    ground_truth,
):
    scores = []

    for class_idx in range(
        len(STRUCTURES)
    ):

        score = binary_dice(
            prediction[class_idx],
            ground_truth[class_idx],
        )

        if score is not None:
            scores.append(
                score
            )

    if not scores:
        return np.nan

    return float(
        np.mean(scores)
    )


def mean_br2_dice(
    prediction,
    ground_truth,
):
    scores = []

    for class_idx in BR2_INDICES:

        score = binary_dice(
            prediction[class_idx],
            ground_truth[class_idx],
        )

        if score is not None:
            scores.append(
                score
            )

    if not scores:
        return None

    return float(
        np.mean(scores)
    )


# ============================================================
# Presence-result files
# ============================================================

def presence_csv_path(
    experiment,
    fold,
):
    fold_root = (
        EXPERIMENTS[experiment]["root"]
        / f"fold_{fold}"
    )

    if experiment == "anatomy":
        return (
            fold_root
            / "test_presence_per_image.csv"
        )

    return (
        fold_root
        / "independent_test"
        / "test_presence_per_image.csv"
    )


def load_all_presence_results():
    frames = []

    for fold in range(5):

        for experiment in EXPERIMENTS:

            path = presence_csv_path(
                experiment,
                fold,
            )

            df = pd.read_csv(
                path,
                dtype={
                    "image_id": str
                },
            )

            df["image_id"] = (
                df["image_id"]
                .map(norm)
            )

            df["fold"] = fold
            df["experiment"] = (
                experiment
            )

            frames.append(
                df
            )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# PASS 1
#
# Scan all test images but store ONLY numbers.
# This avoids the previous RAM problem.
# ============================================================

def collect_selection_metrics(
    device,
):
    records = {}

    for fold in range(5):

        print()
        print("=" * 70)
        print(
            f"Processing fold {fold}"
        )
        print("=" * 70)

        loader, cfg = (
            build_test_loader(
                fold
            )
        )

        models = {}

        for experiment in EXPERIMENTS:

            print(
                "Loading "
                + EXPERIMENTS[
                    experiment
                ]["label"]
                + "..."
            )

            models[experiment] = (
                load_model(
                    experiment,
                    fold,
                    device,
                )
            )

        patch_size = int(
            cfg["patch_size"]
        )

        stride = int(
            cfg["stride"]
        )

        threshold = float(
            cfg["threshold"]
        )

        total = len(
            loader.dataset
        )

        for n, batch in enumerate(
            loader,
            start=1,
        ):

            image = batch[
                "image"
            ][0]

            ground_truth_tensor = batch[
                "mask"
            ][0]

            image_id = norm(
                batch[
                    "image_id"
                ][0]
            )

            ground_truth = (
                ground_truth_tensor
                .detach()
                .cpu()
                .numpy()
            )

            record = {
                "fold": fold,
                "image_id": image_id,
                "mean_dice": {},
                "br2_dice": {},
            }

            print(
                f"\r"
                f"  inference "
                f"{n:3d}/{total} "
                f"image={image_id}",
                end="",
                flush=True,
            )

            for experiment in EXPERIMENTS:

                model = models[
                    experiment
                ]

                with torch.inference_mode():

                    probabilities = (
                        sliding_window_predict(
                            model=model,
                            image=image,
                            device=device,
                            patch_size=patch_size,
                            stride=stride,
                        )
                    )

                prediction = (
                    probabilities
                    .detach()
                    .cpu()
                    .numpy()
                    >= threshold
                ).astype(
                    np.uint8
                )

                record[
                    "mean_dice"
                ][experiment] = (
                    mean_present_dice(
                        prediction,
                        ground_truth,
                    )
                )

                record[
                    "br2_dice"
                ][experiment] = (
                    mean_br2_dice(
                        prediction,
                        ground_truth,
                    )
                )

                del probabilities
                del prediction

            records[
                (
                    fold,
                    image_id,
                )
            ] = record

            del image
            del ground_truth_tensor
            del ground_truth
            del batch

        print()

        for model in models.values():
            del model

        del models
        del loader

        if device.type == "mps":
            torch.mps.empty_cache()

        elif device.type == "cuda":
            torch.cuda.empty_cache()

    return records


# ============================================================
# Case selection
# ============================================================

def select_representative_case(
    records,
    excluded,
):
    """
    Select case closest to median Anatomy-aware whole-image Dice.
    """

    candidates = []

    for key, record in records.items():

        if key in excluded:
            continue

        score = record[
            "mean_dice"
        ]["anatomy"]

        if np.isfinite(score):
            candidates.append(
                (
                    key,
                    score,
                )
            )

    scores = np.array(
        [
            x[1]
            for x in candidates
        ]
    )

    median_score = float(
        np.median(scores)
    )

    candidates.sort(
        key=lambda x:
        abs(
            x[1]
            - median_score
        )
    )

    return candidates[0][0]


def select_br2_improvement_case(
    records,
    excluded,
):
    """
    Select a Br2-positive image where Anatomy-aware gives
    greater Br2 Dice than Baseline.

    To avoid an absurdly poor Baseline extreme case,
    require Anatomy-aware Br2 Dice >= 0.40.
    """

    candidates = []

    for key, record in records.items():

        if key in excluded:
            continue

        baseline = record[
            "br2_dice"
        ]["baseline"]

        anatomy = record[
            "br2_dice"
        ]["anatomy"]

        if (
            baseline is None
            or anatomy is None
        ):
            continue

        if anatomy < 0.40:
            continue

        improvement = (
            anatomy
            - baseline
        )

        if improvement <= 0:
            continue

        candidates.append(
            (
                key,
                improvement,
                anatomy,
                baseline,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No suitable Br2 improvement case found."
        )

    candidates.sort(
        key=lambda x:
        x[1],
        reverse=True,
    )

    return candidates[0][0]


def select_fp_suppression_case(
    presence_results,
    records,
    excluded,
):
    """
    Prefer an image where:
    - Br2 is not annotated;
    - Baseline AND Anatomy-aware predict a Br2 false positive;
    - Class-specific Br2 does not predict it.
    """

    br2 = presence_results[
        presence_results[
            "structure"
        ].isin(
            [
                "Br2a",
                "Br2b",
            ]
        )
    ].copy()

    table = br2.pivot_table(
        index=[
            "fold",
            "image_id",
            "structure",
        ],
        columns="experiment",
        values=[
            "gt_present",
            "pred_present",
        ],
        aggfunc="first",
    )

    table.columns = [
        f"{metric}_{experiment}"
        for metric, experiment
        in table.columns
    ]

    table = (
        table
        .reset_index()
    )

    candidates = []

    for (
        fold,
        image_id,
    ), group in table.groupby(
        [
            "fold",
            "image_id",
        ]
    ):

        key = (
            int(fold),
            norm(image_id),
        )

        if (
            key in excluded
            or key not in records
        ):
            continue

        score = 0

        for _, row in group.iterrows():

            gt = int(
                row[
                    "gt_present_baseline"
                ]
            )

            baseline_pred = int(
                row[
                    "pred_present_baseline"
                ]
            )

            anatomy_pred = int(
                row[
                    "pred_present_anatomy"
                ]
            )

            class_pred = int(
                row[
                    "pred_present_class_specific"
                ]
            )

            if (
                gt == 0
                and class_pred == 0
            ):
                score += (
                    baseline_pred
                    + anatomy_pred
                )

        if score > 0:
            candidates.append(
                (
                    key,
                    score,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No FP suppression example found."
        )

    candidates.sort(
        key=lambda x:
        x[1],
        reverse=True,
    )

    return candidates[0][0]


def select_challenging_case(
    records,
    excluded,
):
    """
    Select low-performing case based on average whole-image
    segmentation Dice across all three configurations.
    """

    candidates = []

    for key, record in records.items():

        if key in excluded:
            continue

        values = [
            record["mean_dice"][
                "baseline"
            ],
            record["mean_dice"][
                "anatomy"
            ],
            record["mean_dice"][
                "class_specific"
            ],
        ]

        average_score = float(
            np.mean(values)
        )

        candidates.append(
            (
                key,
                average_score,
            )
        )

    candidates.sort(
        key=lambda x:
        x[1]
    )

    return candidates[0][0]


# ============================================================
# Save selected-case metrics
# ============================================================

def save_selection_table(
    cases,
    records,
):
    rows = []

    for case_name, key in cases:

        record = records[
            key
        ]

        row = {
            "case": case_name,
            "fold": key[0],
            "image_id": key[1],
        }

        for experiment in EXPERIMENTS:

            row[
                f"{experiment}_mean_dice"
            ] = record[
                "mean_dice"
            ][experiment]

            row[
                f"{experiment}_br2_dice"
            ] = record[
                "br2_dice"
            ][experiment]

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    path = (
        OUTPUT_DIR
        / "figure9_selected_cases.csv"
    )

    df.to_csv(
        path,
        index=False,
    )

    print()
    print(
        "Selected cases:"
    )

    print(
        df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved selection table: {path}"
    )


# ============================================================
# PASS 2
#
# Re-run inference ONLY for the four selected images.
# ============================================================

def recover_selected_images(
    cases,
    device,
):
    selected = {}

    cases_by_fold = {}

    for case_name, key in cases:

        fold, image_id = key

        cases_by_fold.setdefault(
            fold,
            set(),
        ).add(
            image_id
        )

    for fold, wanted_ids in (
        cases_by_fold.items()
    ):

        print()
        print(
            f"Recovering selected images from fold {fold}"
        )

        loader, cfg = (
            build_test_loader(
                fold
            )
        )

        models = {}

        for experiment in EXPERIMENTS:

            models[experiment] = (
                load_model(
                    experiment,
                    fold,
                    device,
                )
            )

        patch_size = int(
            cfg["patch_size"]
        )

        stride = int(
            cfg["stride"]
        )

        threshold = float(
            cfg["threshold"]
        )

        for batch in loader:

            image_id = norm(
                batch[
                    "image_id"
                ][0]
            )

            if image_id not in wanted_ids:
                continue

            print(
                f"  recovering {image_id}"
            )

            image = batch[
                "image"
            ][0]

            ground_truth = (
                batch[
                    "mask"
                ][0]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.uint8
                )
            )

            record = {
                "image": (
                    image.detach()
                    .cpu()
                    .clone()
                ),
                "ground_truth": ground_truth,
                "predictions": {},
            }

            for experiment in EXPERIMENTS:

                model = models[
                    experiment
                ]

                with torch.inference_mode():

                    probabilities = (
                        sliding_window_predict(
                            model=model,
                            image=image,
                            device=device,
                            patch_size=patch_size,
                            stride=stride,
                        )
                    )

                prediction = (
                    probabilities
                    .detach()
                    .cpu()
                    .numpy()
                    >= threshold
                ).astype(
                    np.uint8
                )

                record[
                    "predictions"
                ][experiment] = (
                    prediction
                )

                del probabilities

            selected[
                (
                    fold,
                    image_id,
                )
            ] = record

        for model in models.values():
            del model

        del models
        del loader

        if device.type == "mps":
            torch.mps.empty_cache()

        elif device.type == "cuda":
            torch.cuda.empty_cache()

    return selected


# ============================================================
# Figure creation
# ============================================================

def create_figure(
    cases,
    selected_images,
):
    number_rows = len(
        cases
    )

    fig, axes = plt.subplots(
        number_rows,
        5,
        figsize=(
            18,
            14,
        ),
    )

    titles = [
        "Image",
        "Ground truth",
        "Baseline",
        "Anatomy-aware",
        "Class-specific Br2",
    ]

    for column, title in enumerate(
        titles
    ):

        axes[
            0,
            column,
        ].set_title(
            title,
            fontsize=12,
            fontweight="bold",
        )

    for row, (
        case_name,
        key,
    ) in enumerate(
        cases
    ):

        record = selected_images[
            key
        ]

        image = normalize_image(
            record[
                "image"
            ]
        )

        gt_overlay = (
            make_colored_overlay(
                image,
                record[
                    "ground_truth"
                ],
            )
        )

        baseline_overlay = (
            make_colored_overlay(
                image,
                record[
                    "predictions"
                ]["baseline"],
            )
        )

        anatomy_overlay = (
            make_colored_overlay(
                image,
                record[
                    "predictions"
                ]["anatomy"],
            )
        )

        class_overlay = (
            make_colored_overlay(
                image,
                record[
                    "predictions"
                ]["class_specific"],
            )
        )

        panels = [
            image,
            gt_overlay,
            baseline_overlay,
            anatomy_overlay,
            class_overlay,
        ]

        for column, panel in enumerate(
            panels
        ):

            axes[
                row,
                column,
            ].imshow(
                panel
            )

            axes[
                row,
                column,
            ].axis(
                "off"
            )

        axes[
            row,
            0,
        ].text(
            -0.04,
            0.5,
            case_name,
            transform=axes[
                row,
                0,
            ].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
        )

    # --------------------------------------------------------
    # One common structure legend
    # --------------------------------------------------------

    legend_handles = []

    for structure in STRUCTURES:

        legend_handles.append(
            Patch(
                facecolor=COLORS[
                    structure
                ],
                edgecolor="none",
                label=structure,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=9,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.01,
        ),
    )

    plt.tight_layout(
        rect=[
            0.03,
            0.07,
            1.0,
            1.0,
        ]
    )

    png_path = (
        OUTPUT_DIR
        / "figure9_qualitative_results.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / "figure9_qualitative_results.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print(
        f"Saved PNG: {png_path}"
    )

    print(
        f"Saved PDF: {pdf_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    elif (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):

        device = torch.device(
            "mps"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Existing presence results
    # --------------------------------------------------------

    presence_results = (
        load_all_presence_results()
    )

    # --------------------------------------------------------
    # PASS 1
    # --------------------------------------------------------

   # --------------------------------------------------------
# Reuse already selected Figure 9 cases
# --------------------------------------------------------

selection_csv = (
    OUTPUT_DIR
    / "figure9_selected_cases.csv"
)

if not selection_csv.exists():
    raise FileNotFoundError(
        f"Selection file not found: {selection_csv}"
    )

selection_df = pd.read_csv(
    selection_csv,
    dtype={
        "image_id": str
    },
)

selection_df["image_id"] = (
    selection_df["image_id"]
    .map(norm)
)

case_lookup = {
    "A. Representative": "A",
    "B. Br2 present": "B",
    "C. FP suppression": "C",
    "D. Challenging": "D",
}

cases = []

for _, row in selection_df.iterrows():

    old_label = str(
        row["case"]
    )

    new_label = case_lookup.get(
        old_label,
        old_label,
    )

    key = (
        int(row["fold"]),
        norm(row["image_id"]),
    )

    cases.append(
        (
            new_label,
            key,
        )
    )

print()
print("=" * 70)
print(
    "REUSING EXISTING FIGURE 9 CASES"
)
print("=" * 70)

for case_name, (
    fold,
    image_id,
) in cases:

    print(
        f"{case_name:<5}"
        f" fold={fold}"
        f" image={image_id}"
    )

    metrics = (
        collect_selection_metrics(
            device
        )
    )

    print()
    print(
        f"Recovered metrics for {len(metrics)} test records."
    )

    if len(metrics) != 190:

        print(
            "WARNING: expected 190 test records."
        )

    # --------------------------------------------------------
    # Choose four cases
    # --------------------------------------------------------

    used = set()

    br2_case = (
        select_br2_improvement_case(
            metrics,
            used,
        )
    )

    used.add(
        br2_case
    )

    fp_case = (
        select_fp_suppression_case(
            presence_results,
            metrics,
            used,
        )
    )

    used.add(
        fp_case
    )

    representative_case = (
        select_representative_case(
            metrics,
            used,
        )
    )

    used.add(
        representative_case
    )

    challenging_case = (
        select_challenging_case(
            metrics,
            used,
        )
    )

    cases = [
    ("A", representative_case),
    ("B", br2_case),
    ("C", fp_case),
    ("D", challenging_case),
]

    print()
    print("=" * 70)
    print(
        "SELECTED FIGURE 9 CASES"
    )
    print("=" * 70)

    for case_name, (
        fold,
        image_id,
    ) in cases:

        print(
            f"{case_name:<22}"
            f" fold={fold}"
            f" image={image_id}"
        )

    save_selection_table(
        cases,
        metrics,
    )

    # --------------------------------------------------------
    # PASS 2
    # --------------------------------------------------------

    print()
    print(
        "PASS 2: recovering only the four selected images"
    )

    selected_images = (
        recover_selected_images(
            cases,
            device,
        )
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    create_figure(
        cases,
        selected_images,
    )

    print()
    print(
        "Figure 9 generation complete."
    )


if __name__ == "__main__":
    main()