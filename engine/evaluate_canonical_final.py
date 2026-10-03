#!/usr/bin/env python3
"""
Evaluation-only rerun of the 15 frozen final checkpoints against the
current canonical dataset.

NO TRAINING.
NO CHECKPOINT SELECTION.
NO MODIFICATION OF HISTORICAL OUTPUTS.

Expected canonical GT:
    Br2a = 63
    Br2b = 67
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.structure_names import STRUCTURES
from datasets.metadata import Metadata
from engine.dataloaders import build_full_image_loader
from engine.validate_full import validate_full_image
from models.unetpp import build_model


EXPERIMENTS = {
    "baseline_dice_focal": ROOT / "outputs" / "baseline_dice_focal",
    "anatomy_aware_dice_focal": ROOT / "outputs" / "anatomy_aware_dice_focal",
    "class_specific_br2": ROOT / "outputs" / "class_specific_br2",
}

NEW_ROOT = ROOT / "outputs" / "canonical_63_67_evaluation"

EXPECTED_BR2A = 63
EXPECTED_BR2B = 67


def norm(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def exact_test_indices(df, manifest_path):
    manifest = pd.read_csv(manifest_path)

    required = {"split", "image_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(
            f"{manifest_path} missing columns: {sorted(missing)}"
        )

    metadata_ids = df["image_id"].map(norm)

    if metadata_ids.duplicated().any():
        raise RuntimeError("Duplicate image IDs in metadata.")

    lookup = {
        image_id: i
        for i, image_id in enumerate(metadata_ids)
    }

    test_ids = (
        manifest.loc[
            manifest["split"].astype(str).str.lower() == "test",
            "image_id",
        ]
        .map(norm)
        .tolist()
    )

    missing_ids = [x for x in test_ids if x not in lookup]
    if missing_ids:
        raise RuntimeError(
            f"Test IDs missing from metadata: {missing_ids[:20]}"
        )

    return [lookup[x] for x in test_ids]


def resolve_config(exp_dir, fold):
    fold_cfg = exp_dir / f"fold_{fold}" / "config_used.json"

    if fold_cfg.is_file():
        return fold_cfg

    top_cfg = exp_dir / "config_used.json"
    if top_cfg.is_file():
        return top_cfg

    # The final class-specific training script used this public config
    # directly and did not save per-fold config_used.json files.
    if exp_dir.name == "class_specific_br2":
        public_cfg = ROOT / "configs" / "class_specific_br2_5fold.json"
        if public_cfg.is_file():
            return public_cfg

    raise FileNotFoundError(
        f"No saved config found for {exp_dir.name}, fold {fold}"
    )


def main():

    NEW_ROOT.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Establish canonical metadata/config from a real saved run.
    # ----------------------------------------------------------
    reference_cfg = load_json(
        ROOT
        / "outputs"
        / "baseline_dice_focal"
        / "fold_0"
        / "config_used.json"
    )

    metadata_path = ROOT / reference_cfg["metadata_csv"]
    manifest_root = ROOT / reference_cfg["manifest_root"]

    metadata = Metadata(str(metadata_path))
    df = metadata.df

    if len(df) != 190:
        raise RuntimeError(
            f"Expected 190 metadata records, found {len(df)}."
        )

    # ----------------------------------------------------------
    # Evaluate all three experiments.
    # ----------------------------------------------------------
    for exp_name, exp_dir in EXPERIMENTS.items():

        print("\n" + "=" * 78)
        print(f"EXPERIMENT: {exp_name}")
        print("=" * 78)

        exp_out = NEW_ROOT / exp_name

        # Resume safely: a completed experiment is preserved and skipped.
        pooled_dice_file = exp_out / "pooled_per_structure_dice.csv"
        pooled_presence_file = exp_out / "pooled_presence_summary.csv"

        if pooled_dice_file.is_file() and pooled_presence_file.is_file():
            print(
                f"Completed canonical evaluation already exists for "
                f"{exp_name}; preserving and skipping."
            )
            continue

        exp_out.mkdir(parents=True, exist_ok=True)

        all_dice = []
        all_presence = []
        fold_summaries = []

        for fold in range(5):

            print(f"\n--- Fold {fold} ---")

            cfg_path = resolve_config(exp_dir, fold)
            cfg = load_json(cfg_path)

            checkpoint_path = (
                exp_dir
                / f"fold_{fold}"
                / "checkpoints"
                / "best_model.pt"
            )

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint: {checkpoint_path}"
                )

            manifest_path = (
                ROOT
                / "metadata"
                / "final_cv_190"
                / f"fold_{fold}"
                / "split_manifest.csv"
            )

            test_indices = exact_test_indices(
                df,
                manifest_path,
            )

            print(f"Config     : {cfg_path}")
            print(f"Checkpoint : {checkpoint_path}")
            print(f"Test size  : {len(test_indices)}")

            # ----------------------------------------------
            # Build exactly the same model architecture.
            # ----------------------------------------------
            model = build_model()

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
            )

            model = model.to(device)

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
            )

            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

            model.eval()

            # ----------------------------------------------
            # Current canonical dataset + frozen test split.
            # ----------------------------------------------
            test_loader = build_full_image_loader(
                root_dir=str(ROOT / cfg["root_dir"]),
                metadata_csv=str(ROOT / cfg["metadata_csv"]),
                indices=test_indices,
                resize_to=tuple(cfg["resize_to"]),
                num_workers=cfg["full_test_workers"],
            )

            fold_out = exp_out / f"fold_{fold}"
            fold_out.mkdir(parents=True)

            # ----------------------------------------------
            # EXACT existing full-image evaluator.
            #
            # We deliberately do NOT save prediction PNGs.
            # This changes no metric and saves substantial
            # disk/time.
            # ----------------------------------------------
            with torch.no_grad():
                r = validate_full_image(
                    model=model,
                    loader=test_loader,
                    device=device,
                    epoch=None,
                    patch_size=cfg["patch_size"],
                    stride=cfg["stride"],
                    threshold=cfg["threshold"],
                    min_component_pixels=cfg[
                        "min_component_pixels"
                    ],
                    collect_presence=True,
                    save_all_predictions=False,
                    save_dir=None,
                    return_details=True,
                )

            dice_df = pd.DataFrame(
                {
                    "structure": STRUCTURES,
                    "test_dice": r["class_dice"],
                    "test_count": r["class_counts"],
                }
            )

            dice_df.to_csv(
                fold_out / "test_per_structure_dice.csv",
                index=False,
            )

            r["presence_summary"].to_csv(
                fold_out / "test_presence_summary.csv",
                index=False,
            )

            r["presence_per_image"].to_csv(
                fold_out / "test_presence_per_image.csv",
                index=False,
            )

            summary = {
                "experiment": exp_name,
                "fold": fold,
                "checkpoint": str(
                    checkpoint_path.relative_to(ROOT)
                ),
                "checkpoint_epoch_zero_based": int(
                    checkpoint["epoch"]
                ),
                "best_epoch": int(checkpoint["epoch"]) + 1,
                "best_validation_dice": float(
                    checkpoint["checkpoint_score"]
                ),
                "test_mean_dice": float(r["mean_dice"]),
                "n_test_images": len(test_indices),
                "evaluation_only": True,
                "canonical_ground_truth": True,
            }

            save_json(
                summary,
                fold_out / "test_summary.json",
            )

            tmp_dice = dice_df.copy()
            tmp_dice["fold"] = fold
            all_dice.append(tmp_dice)

            tmp_presence = r["presence_summary"].copy()
            tmp_presence["fold"] = fold
            all_presence.append(tmp_presence)

            fold_summaries.append(summary)

            print(
                f"Fold {fold} test mean Dice: "
                f"{r['mean_dice']:.6f}"
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # --------------------------------------------------
        # Pool Dice across independent test folds.
        # Same count-weighted aggregation as paper script.
        # --------------------------------------------------
        dice_all = pd.concat(all_dice, ignore_index=True)

        pooled_dice_rows = []

        for structure in STRUCTURES:
            x = dice_all[
                dice_all["structure"] == structure
            ].copy()

            valid = (
                x["test_dice"].notna()
                & (x["test_count"] > 0)
            )

            x = x.loc[valid]

            total_count = float(x["test_count"].sum())

            if total_count > 0:
                pooled_dice = float(
                    (
                        x["test_dice"]
                        * x["test_count"]
                    ).sum()
                    / total_count
                )
            else:
                pooled_dice = None

            pooled_dice_rows.append(
                {
                    "structure": structure,
                    "test_dice": pooled_dice,
                    "test_count": int(total_count),
                }
            )

        pooled_dice = pd.DataFrame(pooled_dice_rows)

        pooled_dice.to_csv(
            exp_out / "pooled_per_structure_dice.csv",
            index=False,
        )

        # --------------------------------------------------
        # Pool presence confusion counts.
        # --------------------------------------------------
        presence_all = pd.concat(
            all_presence,
            ignore_index=True,
        )

        pooled_presence = (
            presence_all.groupby(
                "structure",
                sort=False,
            )[
                [
                    "gt_present",
                    "pred_present",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                ]
            ]
            .sum()
            .reset_index()
        )

        pooled_presence["precision"] = (
            pooled_presence["tp"]
            / (
                pooled_presence["tp"]
                + pooled_presence["fp"]
            ).replace(0, pd.NA)
        )

        pooled_presence["recall"] = (
            pooled_presence["tp"]
            / (
                pooled_presence["tp"]
                + pooled_presence["fn"]
            ).replace(0, pd.NA)
        )

        pooled_presence["f1"] = (
            2
            * pooled_presence["precision"]
            * pooled_presence["recall"]
            / (
                pooled_presence["precision"]
                + pooled_presence["recall"]
            )
        )

        pooled_presence["specificity"] = (
            pooled_presence["tn"]
            / (
                pooled_presence["tn"]
                + pooled_presence["fp"]
            ).replace(0, pd.NA)
        )

        pooled_presence.to_csv(
            exp_out / "pooled_presence_summary.csv",
            index=False,
        )

        pd.DataFrame(fold_summaries).to_csv(
            exp_out / "fold_test_summary.csv",
            index=False,
        )

        # --------------------------------------------------
        # HARD canonical GT sanity check.
        # --------------------------------------------------
        counts = dict(
            zip(
                pooled_dice["structure"],
                pooled_dice["test_count"],
            )
        )

        presence_counts = dict(
            zip(
                pooled_presence["structure"],
                pooled_presence["gt_present"],
            )
        )

        print("\nCanonical GT sanity check:")
        print(
            f"  Br2a Dice test_count = "
            f"{counts.get('Br2a')}"
        )
        print(
            f"  Br2b Dice test_count = "
            f"{counts.get('Br2b')}"
        )
        print(
            f"  Br2a presence count  = "
            f"{presence_counts.get('Br2a')}"
        )
        print(
            f"  Br2b presence count  = "
            f"{presence_counts.get('Br2b')}"
        )

        if (
            counts.get("Br2a") != EXPECTED_BR2A
            or counts.get("Br2b") != EXPECTED_BR2B
            or presence_counts.get("Br2a") != EXPECTED_BR2A
            or presence_counts.get("Br2b") != EXPECTED_BR2B
        ):
            raise RuntimeError(
                f"{exp_name}: canonical GT sanity check FAILED."
            )

        print(
            f"{exp_name}: canonical 63/67 GT check PASS."
        )

    print("\n" + "=" * 78)
    print("ALL 15 FROZEN CHECKPOINTS EVALUATED SUCCESSFULLY")
    print(f"Results: {NEW_ROOT}")
    print("=" * 78)


if __name__ == "__main__":
    main()
