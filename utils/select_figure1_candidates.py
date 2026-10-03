#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


STRUCTURES = [
    "Br1a", "Br1b", "Br2a", "Br2b",
    "Cb1", "Cb2", "Ch1", "Ch2",
    "Cl1", "Cl2", "D1", "D2",
    "En1", "En2", "Hm1", "Hm2",
    "M1", "M2", "N",
    "Oc1", "Oc2",
    "Op1", "Op2",
    "P", "VC",
]

BR2_STRUCTURES = ["Br2a", "Br2b"]


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read: {path}")
    return img


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, shape):
    if not path.exists():
        return np.zeros(shape, dtype=bool)

    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, dtype=bool)

    if m.shape != shape:
        m = cv2.resize(
            m,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return m > 0


def bbox_from_mask(mask, margin=80):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    h, w = mask.shape

    x1 = max(0, int(xs.min()) - margin)
    x2 = min(w, int(xs.max()) + margin + 1)
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(h, int(ys.max()) + margin + 1)

    return x1, y1, x2, y2


def expand_bbox(bbox, shape, min_size=450):
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    h, w = shape

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    bw = max(x2 - x1, min_size)
    bh = max(y2 - y1, min_size)

    x1 = max(0, cx - bw // 2)
    y1 = max(0, cy - bh // 2)

    x2 = min(w, x1 + bw)
    y2 = min(h, y1 + bh)

    # Correct if clipped at right/bottom
    x1 = max(0, x2 - bw)
    y1 = max(0, y2 - bh)

    return int(x1), int(y1), int(x2), int(y2)


def local_contrast(gray, mask):
    """
    Approximate visibility:
    absolute difference between mean intensity inside the structure
    and a local ring surrounding it.

    Smaller values -> weaker local image contrast.
    """
    if mask.sum() == 0:
        return np.nan

    mask_u8 = mask.astype(np.uint8)

    kernel = np.ones((31, 31), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)

    ring = dilated & (~mask)

    if ring.sum() == 0:
        return np.nan

    inside = gray[mask].astype(np.float32)
    outside = gray[ring].astype(np.float32)

    return float(abs(inside.mean() - outside.mean()))


def contour_overlay(rgb, masks, thickness=3):
    out = rgb.copy()

    colors = {
        "Br2a": (255, 0, 255),   # magenta
        "Br2b": (0, 255, 255),   # cyan
    }

    default_colors = [
        (255, 64, 64),
        (64, 255, 64),
        (64, 64, 255),
        (255, 180, 0),
        (180, 0, 255),
        (0, 180, 255),
    ]

    color_idx = 0

    for name, mask in masks.items():
        if mask.sum() == 0:
            continue

        color = colors.get(
            name,
            default_colors[color_idx % len(default_colors)]
        )

        if name not in colors:
            color_idx += 1

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            out,
            contours,
            -1,
            color,
            thickness,
        )

    return out


def pairwise_overlap_score(masks):
    """
    Returns maximum pixel overlap between any two anatomical masks.
    """
    names = [k for k, v in masks.items() if v.sum() > 0]

    best_overlap = 0
    best_pair = None
    best_intersection = None

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]

            inter = masks[a] & masks[b]
            n = int(inter.sum())

            if n > best_overlap:
                best_overlap = n
                best_pair = (a, b)
                best_intersection = inter

    return best_overlap, best_pair, best_intersection


def br2_expected_region(masks, shape):
    """
    Creates an approximate region for a GT-empty Br2 example using Br1 masks.
    This is only for visualization/cropping, not quantitative analysis.
    """
    combined = np.zeros(shape, dtype=bool)

    for name in ["Br1a", "Br1b"]:
        combined |= masks[name]

    if combined.sum() == 0:
        return None

    kernel = np.ones((121, 121), np.uint8)

    region = cv2.dilate(
        combined.astype(np.uint8),
        kernel,
        iterations=1
    ).astype(bool)

    return region


def scan_dataset(ventral_dir):
    records = []

    folders = sorted(
        p for p in ventral_dir.iterdir()
        if p.is_dir() and (p / "image.png").exists()
    )

    print(f"Found {len(folders)} image folders")

    for idx, folder in enumerate(folders, 1):

        image_id = folder.name

        gray = load_gray(folder / "image.png")
        shape = gray.shape

        masks = {
            s: load_mask(folder / f"{s}.png", shape)
            for s in STRUCTURES
        }

        areas = {
            s: int(m.sum())
            for s, m in masks.items()
        }

        n_present = sum(v > 0 for v in areas.values())

        total_foreground = int(
            np.logical_or.reduce(list(masks.values())).sum()
        )

        br2_present = (
            areas["Br2a"] > 0 or
            areas["Br2b"] > 0
        )

        br2_total_area = (
            areas["Br2a"] +
            areas["Br2b"]
        )

        contrasts = []

        for s in BR2_STRUCTURES:
            if areas[s] > 0:
                c = local_contrast(gray, masks[s])

                if np.isfinite(c):
                    contrasts.append(c)

        br2_contrast = (
            float(np.mean(contrasts))
            if contrasts
            else np.nan
        )

        overlap_pixels, overlap_pair, overlap_mask = pairwise_overlap_score(
            masks
        )

        records.append({
            "image_id": image_id,
            "folder": folder,
            "n_present": n_present,
            "total_foreground": total_foreground,
            "br2_present": br2_present,
            "br2_area": br2_total_area,
            "br2_contrast": br2_contrast,
            "overlap_pixels": overlap_pixels,
            "overlap_pair": overlap_pair,
        })

        if idx % 25 == 0:
            print(f"  scanned {idx}/{len(folders)}")

    return records


def get_candidates(records, top_k):

    # ---------------------------------------------------------
    # A: representative / clear anatomy
    #
    # Prefer many annotated structures and substantial foreground.
    # ---------------------------------------------------------
    rep = sorted(
        records,
        key=lambda r: (
            r["n_present"],
            r["total_foreground"],
        ),
        reverse=True,
    )[:top_k]

    # ---------------------------------------------------------
    # B: difficult / weakly visible Br2
    #
    # Require Br2 GT and rank primarily by low local contrast.
    # Smaller Br2 is used as secondary criterion.
    # ---------------------------------------------------------
    br2_positive = [
        r for r in records
        if r["br2_present"]
        and np.isfinite(r["br2_contrast"])
    ]

    weak = sorted(
        br2_positive,
        key=lambda r: (
            r["br2_contrast"],
            r["br2_area"],
        )
    )[:top_k]

    # ---------------------------------------------------------
    # C: no Br2 reference annotation
    #
    # Prefer examples with many surrounding structures present,
    # so expected Br2 anatomy is easier to interpret visually.
    # ---------------------------------------------------------
    no_br2 = [
        r for r in records
        if not r["br2_present"]
    ]

    absent = sorted(
        no_br2,
        key=lambda r: (
            r["n_present"],
            r["total_foreground"],
        ),
        reverse=True,
    )[:top_k]

    # ---------------------------------------------------------
    # D: anatomical overlap
    # ---------------------------------------------------------
    overlapping = [
        r for r in records
        if r["overlap_pixels"] > 0
    ]

    overlap = sorted(
        overlapping,
        key=lambda r: r["overlap_pixels"],
        reverse=True,
    )[:top_k]

    return {
        "A_clear": rep,
        "B_weak_Br2": weak,
        "C_no_Br2_annotation": absent,
        "D_overlap": overlap,
    }


def make_candidate_sheet(category, records, output_path):

    n = len(records)

    if n == 0:
        print(f"No candidates for {category}")
        return

    cols = min(5, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.2 * cols, 4.2 * rows),
    )

    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, record in zip(axes, records):

        folder = record["folder"]

        rgb = load_rgb(folder / "image.png")
        shape = rgb.shape[:2]

        masks = {
            s: load_mask(folder / f"{s}.png", shape)
            for s in STRUCTURES
        }

        # Determine crop
        if category == "B_weak_Br2":
            roi = masks["Br2a"] | masks["Br2b"]

        elif category == "C_no_Br2_annotation":
            roi = br2_expected_region(masks, shape)

            if roi is None:
                roi = np.logical_or.reduce(list(masks.values()))

        elif category == "D_overlap":
            _, pair, inter = pairwise_overlap_score(masks)

            if inter is not None:
                roi = inter

                display_masks = {
                    pair[0]: masks[pair[0]],
                    pair[1]: masks[pair[1]],
                }
            else:
                roi = np.logical_or.reduce(list(masks.values()))
                display_masks = masks

        else:
            roi = np.logical_or.reduce(list(masks.values()))

        bbox = bbox_from_mask(roi, margin=100)
        bbox = expand_bbox(bbox, shape, min_size=500)

        if bbox is None:
            x1, y1 = 0, 0
            x2, y2 = shape[1], shape[0]
        else:
            x1, y1, x2, y2 = bbox

        if category == "B_weak_Br2":
            display_masks = {
                "Br2a": masks["Br2a"],
                "Br2b": masks["Br2b"],
            }

        elif category == "C_no_Br2_annotation":
            display_masks = {
                "Br1a": masks["Br1a"],
                "Br1b": masks["Br1b"],
            }

        elif category != "D_overlap":
            display_masks = masks

        overlay = contour_overlay(
            rgb,
            display_masks,
            thickness=3,
        )

        crop = overlay[y1:y2, x1:x2]

        ax.imshow(crop)

        if category == "B_weak_Br2":
            title = (
                f"{record['image_id']}\n"
                f"Br2 area={record['br2_area']} px | "
                f"contrast={record['br2_contrast']:.2f}"
            )

        elif category == "D_overlap":
            pair = record["overlap_pair"]

            title = (
                f"{record['image_id']}\n"
                f"{pair[0]}–{pair[1]} overlap: "
                f"{record['overlap_pixels']} px"
            )

        else:
            title = (
                f"{record['image_id']}\n"
                f"{record['n_present']} structures"
            )

        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        category.replace("_", " "),
        fontsize=15,
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def write_csv(candidate_groups, output_csv):

    with open(output_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "category",
            "rank",
            "image_id",
            "n_present",
            "br2_area_pixels",
            "br2_local_contrast",
            "max_overlap_pixels",
            "overlap_pair",
        ])

        for category, records in candidate_groups.items():

            for rank, r in enumerate(records, 1):

                pair = (
                    "-".join(r["overlap_pair"])
                    if r["overlap_pair"]
                    else ""
                )

                writer.writerow([
                    category,
                    rank,
                    r["image_id"],
                    r["n_present"],
                    r["br2_area"],
                    r["br2_contrast"],
                    r["overlap_pixels"],
                    pair,
                ])


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ventral-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figure1_candidates"),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = scan_dataset(args.ventral_dir)

    candidates = get_candidates(
        records,
        args.top_k,
    )

    for category, recs in candidates.items():

        print(f"\n{category}")

        for i, r in enumerate(recs, 1):

            print(
                f"{i:2d}. {r['image_id']} | "
                f"structures={r['n_present']} | "
                f"Br2 area={r['br2_area']} | "
                f"contrast={r['br2_contrast']} | "
                f"overlap={r['overlap_pixels']} "
                f"{r['overlap_pair']}"
            )

        make_candidate_sheet(
            category,
            recs,
            args.output_dir / f"{category}.png",
        )

    write_csv(
        candidates,
        args.output_dir / "figure1_candidates.csv",
    )

    print("\nSaved candidate sheets to:")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()