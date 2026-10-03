#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Canonical structure names
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


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    return mask > 0


def scan_dataset(ventral_dir):

    folders = sorted(
        p for p in ventral_dir.iterdir()
        if p.is_dir() and (p / "image.png").exists()
    )

    if not folders:
        raise RuntimeError(
            f"No image-ID folders found in:\n{ventral_dir}"
        )

    print(f"Found {len(folders)} annotation records.")

    # One list of relative areas per structure.
    # Only non-empty masks are added.
    relative_areas = {
        s: []
        for s in STRUCTURES
    }

    occurrence = {
        s: 0
        for s in STRUCTURES
    }

    for i, folder in enumerate(folders, 1):

        image_path = folder / "image.png"

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read {image_path}"
            )

        h, w = image.shape

        image_area = float(h * w)

        for structure in STRUCTURES:

            mask_path = folder / f"{structure}.png"

            if not mask_path.exists():
                # Missing file -> treated as no reference annotation
                continue

            mask = load_mask(mask_path)

            if mask is None:
                continue

            # Safety in case mask dimensions differ.
            if mask.shape != image.shape:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0

            pixels = int(mask.sum())

            if pixels == 0:
                continue

            occurrence[structure] += 1

            relative_area_percent = (
                100.0 * pixels / image_area
            )

            relative_areas[structure].append(
                relative_area_percent
            )

        if i % 25 == 0:
            print(
                f"Processed {i}/{len(folders)}"
            )

    return folders, occurrence, relative_areas


def calculate_statistics(
    folders,
    occurrence,
    relative_areas
):

    n_records = len(folders)

    stats = []

    for structure in STRUCTURES:

        areas = np.asarray(
            relative_areas[structure],
            dtype=float
        )

        n = occurrence[structure]

        occurrence_percent = (
            100.0 * n / n_records
        )

        if len(areas) > 0:

            mean_area = float(
                np.mean(areas)
            )

            median_area = float(
                np.median(areas)
            )

            std_area = float(
                np.std(
                    areas,
                    ddof=1
                )
            ) if len(areas) > 1 else 0.0

            if mean_area > 0:
                cv = std_area / mean_area
            else:
                cv = np.nan

        else:

            mean_area = np.nan
            median_area = np.nan
            std_area = np.nan
            cv = np.nan

        stats.append({
            "structure": structure,
            "occurrence_n": n,
            "occurrence_percent": occurrence_percent,
            "mean_relative_area_percent": mean_area,
            "median_relative_area_percent": median_area,
            "std_relative_area_percent": std_area,
            "cv": cv,
        })

    return stats


# ============================================================
# CSV
# ============================================================

def write_csv(stats, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "structure",
        "occurrence_n",
        "occurrence_percent",
        "mean_relative_area_percent",
        "median_relative_area_percent",
        "std_relative_area_percent",
        "cv",
    ]

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in stats:
            writer.writerow(row)

    print(f"Saved: {output_path}")


# ============================================================
# LaTeX Table 2
# ============================================================

def write_latex_table(stats, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Descriptive statistics of the twenty-five annotated skeletal structures, "
        r"including occurrence frequency, mean relative area expressed as a percentage of "
        r"the original image area, and coefficient of variation (CV). Mean relative area "
        r"and CV were calculated using only images in which the corresponding structure "
        r"was present.}"
    )
    lines.append(
        r"\label{tab:structure_statistics}"
    )
    lines.append(r"\small")
    lines.append(
        r"\begin{tabular}{lrrrr}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"Structure & "
        r"Occurrence (n) & "
        r"Occurrence (\%) & "
        r"Mean relative area (\%) & "
        r"CV \\"
    )
    lines.append(r"\midrule")

    for row in stats:

        structure = row["structure"]
        n = row["occurrence_n"]
        freq = row["occurrence_percent"]
        mean_area = row[
            "mean_relative_area_percent"
        ]
        cv = row["cv"]

        if np.isnan(mean_area):
            mean_str = "--"
        else:
            mean_str = f"{mean_area:.3f}"

        if np.isnan(cv):
            cv_str = "--"
        else:
            cv_str = f"{cv:.3f}"

        lines.append(
            f"{structure} & "
            f"{n:d} & "
            f"{freq:.2f} & "
            f"{mean_str} & "
            f"{cv_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_path.write_text(
        "\n".join(lines) + "\n"
    )

    print(f"Saved: {output_path}")


# ============================================================
# Figure 3
# ============================================================

def make_figure(stats, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    structures = np.array([
        r["structure"]
        for r in stats
    ])

    frequencies = np.array([
        r["occurrence_percent"]
        for r in stats
    ])

    mean_areas = np.array([
        r["mean_relative_area_percent"]
        for r in stats
    ])

    # Sort smallest -> largest so rare/small structures
    # appear at the bottom of each horizontal plot.
    order_freq = np.argsort(frequencies)[::-1]
    order_size = np.argsort(mean_areas)[::-1]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 8)
    )

    # ========================================================
    # A. Annotation frequency
    # ========================================================

    ax = axes[0]

    y = np.arange(len(STRUCTURES))

    values = frequencies[order_freq]
    labels = structures[order_freq]

    bars = ax.barh(
        y,
        values
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=9
    )

    ax.set_xlabel(
        "Annotation frequency (%)",
        fontsize=10
    )

    ax.set_title(
        "A. Annotation frequency",
        fontsize=12,
        fontweight="bold"
    )

    # Extra space for numerical labels
    ax.set_xlim(
        0,
        max(values) * 1.13
    )

    ax.grid(
        axis="x",
        alpha=0.20
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Numerical value at end of bar
    for bar, value in zip(bars, values):

        ax.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5
        )

    # ========================================================
    # B. Mean relative structure area
    # ========================================================

    ax = axes[1]

    values = mean_areas[order_size]
    labels = structures[order_size]

    bars = ax.barh(
        y,
        values
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=9
    )

    ax.set_xlabel(
        "Mean relative mask area (%)",
        fontsize=10
    )

    ax.set_title(
        "B. Structure size",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlim(
        0,
        max(values) * 1.18
    )

    ax.grid(
        axis="x",
        alpha=0.20
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Numerical value at end of bar
    for bar, value in zip(bars, values):

        ax.text(
            bar.get_width() + max(values) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}%",
            va="center",
            ha="left",
            fontsize=8.5
        )

    plt.tight_layout(
        w_pad=3.0
    )

    png_path = (
        output_dir /
        "figure3_dataset_statistics.png"
    )

    pdf_path = (
        output_dir /
        "figure3_dataset_statistics.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# Terminal summary
# ============================================================

def print_statistics(stats, n_records):

    print("\n")
    print(
        f"{'Structure':<10}"
        f"{'n':>7}"
        f"{'%':>10}"
        f"{'Mean area %':>15}"
        f"{'Median area %':>17}"
        f"{'CV':>10}"
    )

    print("-" * 69)

    for row in stats:

        print(
            f"{row['structure']:<10}"
            f"{row['occurrence_n']:>7d}"
            f"{row['occurrence_percent']:>10.2f}"
            f"{row['mean_relative_area_percent']:>15.3f}"
            f"{row['median_relative_area_percent']:>17.3f}"
            f"{row['cv']:>10.3f}"
        )

    print("\n")
    print(
        f"Total annotation records: {n_records}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 3 and Table 2 directly "
            "from the ventral skeletal annotation masks."
        )
    )

    parser.add_argument(
        "--ventral-dir",
        type=Path,
        default=Path("ventral"),
        help="Default: ./ventral",
    )

    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
        help="Default: ./figures",
    )

    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("tables"),
        help="Default: ./tables",
    )

    args = parser.parse_args()

    (
        folders,
        occurrence,
        relative_areas,
    ) = scan_dataset(
        args.ventral_dir
    )

    stats = calculate_statistics(
        folders,
        occurrence,
        relative_areas,
    )

    print_statistics(
        stats,
        len(folders)
    )

    write_csv(
        stats,
        args.table_dir /
        "table2_structure_statistics.csv"
    )

    write_latex_table(
        stats,
        args.table_dir /
        "table2_structure_statistics.tex"
    )

    make_figure(
        stats,
        args.figure_dir
    )


if __name__ == "__main__":
    main()