#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Full five-fold 3/1/1 evaluation with anatomy-aware patch sampling.

For each fold:
    3 fixed fish-level folds -> training
    1 fixed fish-level fold  -> validation/checkpoint selection
    1 fixed fish-level fold  -> independent test evaluation once

The frozen corrected 190-record fish-level split manifests are reused. No splitter
class is imported or regenerated. Relative to the final Dice+Focal baseline,
the methodological change is the anatomy-aware training sampler.

Expected manifests:
    metadata/final_cv_190/fold_0/split_manifest.csv
    ...
    metadata/final_cv_190/fold_4/split_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.structure_names import STRUCTURES
from datasets.metadata import Metadata
from engine.dataloaders import build_full_image_loader, build_patch_loaders
from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from engine.validate_full import validate_full_image
from losses.combined import CombinedLoss
from models.unetpp import build_model
from utils.save_metrics import save_metrics_csv
from utils.seed import seed_everything


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def normalize_id(value: Any) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0, seconds))))


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def exact_split_from_manifest(
    metadata_df: pd.DataFrame,
    manifest_path: Path,
) -> tuple[list[int], list[int], list[int]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = {"split", "image_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(
            f"{manifest_path} is missing columns: {sorted(missing)}. "
            f"Available: {manifest.columns.tolist()}"
        )

    metadata_ids = metadata_df["image_id"].map(normalize_id)
    if metadata_ids.duplicated().any():
        duplicates = sorted(
            metadata_ids[metadata_ids.duplicated(keep=False)].unique()
        )
        raise ValueError(f"Duplicate image IDs in metadata: {duplicates[:20]}")

    lookup = {
        image_id: index
        for index, image_id in enumerate(metadata_ids)
    }

    result: dict[str, list[int]] = {}
    for split_name in ("train", "val", "test"):
        image_ids = (
            manifest.loc[
                manifest["split"].astype(str).str.lower() == split_name,
                "image_id",
            ]
            .map(normalize_id)
            .tolist()
        )
        if not image_ids:
            raise ValueError(
                f"No {split_name} rows found in manifest: {manifest_path}"
            )
        absent = [image_id for image_id in image_ids if image_id not in lookup]
        if absent:
            raise ValueError(
                f"{split_name} image IDs absent from metadata: {absent[:20]}"
            )
        result[split_name] = [lookup[image_id] for image_id in image_ids]

    train_fish = set(metadata_df.iloc[result["train"]]["fish_id"])
    val_fish = set(metadata_df.iloc[result["val"]]["fish_id"])
    test_fish = set(metadata_df.iloc[result["test"]]["fish_id"])

    if not train_fish.isdisjoint(val_fish):
        raise RuntimeError("Fish leakage between training and validation.")
    if not train_fish.isdisjoint(test_fish):
        raise RuntimeError("Fish leakage between training and test.")
    if not val_fish.isdisjoint(test_fish):
        raise RuntimeError("Fish leakage between validation and test.")

    all_indices = (
        result["train"] + result["val"] + result["test"]
    )
    if len(all_indices) != len(set(all_indices)):
        raise RuntimeError("Image leakage between train/val/test subsets.")

    return result["train"], result["val"], result["test"]


def save_checkpoint(
    path: Path,
    fold: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    score: float,
) -> None:
    torch.save(
        {
            "fold": fold,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "checkpoint_score": score,
        },
        path,
    )


def load_best_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return checkpoint


def presence_metrics_for_targets(
    presence_summary: pd.DataFrame | None,
    targets: list[str],
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    if presence_summary is None:
        return metrics

    for target in targets:
        row = presence_summary.loc[
            presence_summary["structure"] == target
        ]
        if len(row) != 1:
            continue

        row = row.iloc[0]
        tp = int(row["tp"])
        fp = int(row["fp"])
        fn = int(row["fn"])

        metrics[f"{target}_tp"] = tp
        metrics[f"{target}_fp"] = fp
        metrics[f"{target}_fn"] = fn
        metrics[f"{target}_precision"] = tp / max(tp + fp, 1)
        metrics[f"{target}_recall"] = tp / max(tp + fn, 1)
        metrics[f"{target}_f1"] = (
            2 * tp / max(2 * tp + fp + fn, 1)
        )

    return metrics


def print_target_presence(
    title: str,
    presence_summary: pd.DataFrame | None,
    targets: list[str],
) -> None:
    if presence_summary is None:
        return

    print(title, flush=True)
    for target in targets:
        row = presence_summary.loc[
            presence_summary["structure"] == target
        ]
        if len(row) != 1:
            continue

        row = row.iloc[0]
        tp = int(row["tp"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)

        print(
            f"  {target}: TP={tp} FP={fp} FN={fn} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}",
            flush=True,
        )


def preflight(cfg: dict[str, Any], metadata_df: pd.DataFrame) -> None:
    root_dir = resolve_repo_path(cfg["root_dir"])
    metadata_csv = resolve_repo_path(cfg["metadata_csv"])
    manifest_root = resolve_repo_path(cfg["baseline_manifest_root"])

    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    print("\nPreflight checks", flush=True)
    print(f"Repository       : {REPO_ROOT}", flush=True)
    print(f"Dataset root     : {root_dir}", flush=True)
    print(f"Metadata         : {metadata_csv}", flush=True)
    print(f"Images in metadata: {len(metadata_df)}", flush=True)

    if len(metadata_df) != 190:
        raise RuntimeError(f"Expected 190 metadata rows, found {len(metadata_df)}.")
    if "image_id" not in metadata_df.columns or "fish_id" not in metadata_df.columns:
        raise KeyError("Metadata must contain image_id and fish_id.")
    metadata_ids = metadata_df["image_id"].map(normalize_id)
    if metadata_ids.duplicated().any():
        raise RuntimeError("Duplicate image_id values found in metadata.")
    if metadata_df["fish_id"].nunique() != 34:
        raise RuntimeError(
            f"Expected 34 biological fish, found {metadata_df['fish_id'].nunique()}."
        )
    excluded = {"525410710", "503027953"}
    bad = sorted(set(metadata_ids) & excluded)
    if bad:
        raise RuntimeError(f"Excluded image IDs are still present: {bad}")

    expected_counts = {
        0: (115, 37, 38, 21, 6, 7),
        1: (115, 38, 37, 21, 7, 6),
        2: (113, 39, 38, 20, 7, 7),
        3: (113, 38, 39, 20, 7, 7),
        4: (114, 38, 38, 20, 7, 7),
    }
    test_ids_across_folds: list[str] = []

    for fold in range(int(cfg["n_splits"])):
        manifest_path = (
            manifest_root / f"fold_{fold}" / "split_manifest.csv"
        )
        train_idx, val_idx, test_idx = exact_split_from_manifest(
            metadata_df, manifest_path
        )
        test_ids_across_folds.extend(
            metadata_df.iloc[test_idx]["image_id"].map(normalize_id).tolist()
        )
        print(
            f"Fold {fold}: train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}",
            flush=True,
        )

    if len(test_ids_across_folds) != len(metadata_df):
        raise RuntimeError(
            "Across the five manifests, the total number of test assignments "
            f"is {len(test_ids_across_folds)}, expected {len(metadata_df)}."
        )

    if len(set(test_ids_across_folds)) != len(metadata_df):
        raise RuntimeError(
            "Across the five manifests, each image must appear exactly once "
            "as an independent test image."
        )

    print(
        "All images appear exactly once across the five independent test sets.",
        flush=True,
    )


def run_fold(
    fold: int,
    cfg: dict[str, Any],
    metadata_df: pd.DataFrame,
) -> dict[str, Any]:
    output_root = resolve_repo_path(cfg["output_root"])
    output_dir = output_root / f"fold_{fold}"
    checkpoint_dir = output_dir / "checkpoints"
    test_prediction_dir = output_dir / "test_predictions"
    complete_marker = output_dir / "FOLD_COMPLETE.marker"

    if complete_marker.exists() and not cfg.get("rerun_completed_folds", False):
        summary_path = output_dir / "fold_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(
                f"Fold {fold} is marked complete but summary is missing."
            )
        print(f"\nFold {fold} already complete; skipping.", flush=True)
        return load_json(summary_path)

    if output_dir.exists() and cfg.get("clean_incomplete_fold", False):
        shutil.rmtree(output_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    test_prediction_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        resolve_repo_path(cfg["baseline_manifest_root"])
        / f"fold_{fold}"
        / "split_manifest.csv"
    )

    train_idx, val_idx, test_idx = exact_split_from_manifest(
        metadata_df, manifest_path
    )

    fold_cfg = json.loads(json.dumps(cfg))
    fold_cfg["fold"] = fold
    fold_cfg["baseline_split_manifest"] = str(manifest_path)
    fold_cfg["sampler"]["samples_per_epoch"] = len(train_idx)
    fold_cfg["sampler"]["seed"] = int(cfg.get("seed", 42)) + fold
    save_json(fold_cfg, output_dir / "config_used.json")

    # Preserve the exact split beside the new results.
    shutil.copy2(manifest_path, output_dir / "split_manifest.csv")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 78, flush=True)
    print(
        f"ANATOMY-AWARE CV ITERATION {fold + 1}/{cfg['n_splits']} | "
        f"Loss: Dice+Focal",
        flush=True,
    )
    print(
        f"Images -> train: {len(train_idx)} | val: {len(val_idx)} | "
        f"test: {len(test_idx)} | device: {device}",
        flush=True,
    )
    print(
        f"Sampler -> positive: {fold_cfg['sampler']['target_positive_prob']:.2f} | "
        f"hard negative: {fold_cfg['sampler']['target_hard_negative_prob']:.2f} | "
        f"general foreground: {fold_cfg['sampler']['general_foreground_prob']:.2f} | "
        f"random: {fold_cfg['sampler']['random_prob']:.2f}",
        flush=True,
    )
    print("=" * 78, flush=True)

    train_loader, patch_val_loader = build_patch_loaders(
        root_dir=str(resolve_repo_path(cfg["root_dir"])),
        metadata_csv=str(resolve_repo_path(cfg["metadata_csv"])),
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=int(cfg["batch_size"]),
        patch_size=int(cfg["patch_size"]),
        resize_to=tuple(cfg["resize_to"]),
        num_workers=int(cfg["num_workers"]),
        sampler_config=fold_cfg["sampler"],
    )

    full_val_loader = build_full_image_loader(
        root_dir=str(resolve_repo_path(cfg["root_dir"])),
        metadata_csv=str(resolve_repo_path(cfg["metadata_csv"])),
        indices=val_idx,
        resize_to=tuple(cfg["resize_to"]),
        num_workers=int(cfg["full_valid_workers"]),
    )

    full_test_loader = build_full_image_loader(
        root_dir=str(resolve_repo_path(cfg["root_dir"])),
        metadata_csv=str(resolve_repo_path(cfg["metadata_csv"])),
        indices=test_idx,
        resize_to=tuple(cfg["resize_to"]),
        num_workers=int(cfg["full_test_workers"]),
    )

    model = build_model().to(device)
    criterion = CombinedLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["epochs"]),
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    writer = SummaryWriter(str(output_dir / "tensorboard"))

    best_score = float("-inf")
    best_epoch = -1
    fold_start = time.perf_counter()
    targets = list(cfg["sampler"]["target_structures"])

    for epoch in range(int(cfg["epochs"])):
        epoch_start = time.perf_counter()

        print("\n" + "-" * 78, flush=True)
        print(
            f"Fold {fold + 1}/{cfg['n_splits']} | "
            f"Epoch {epoch + 1}/{cfg['epochs']}",
            flush=True,
        )
        print("-" * 78, flush=True)

        print("[1/3] Patch-based training...", flush=True)
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        print("[2/3] Patch validation (monitoring only)...", flush=True)
        (
            patch_val_loss,
            patch_mean_dice,
            patch_class_dice,
            patch_counts,
        ) = validate(
            model=model,
            loader=patch_val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        run_full_val = (
            epoch == 0
            or (epoch + 1) % int(cfg["full_validation_every"]) == 0
        )

        full_val = None
        if run_full_val:
            print(
                "[3/3] Full-image validation "
                "(checkpoint selection)...",
                flush=True,
            )
            full_val = validate_full_image(
                model=model,
                loader=full_val_loader,
                device=device,
                epoch=epoch,
                patch_size=int(cfg["patch_size"]),
                stride=int(cfg["stride"]),
                threshold=float(cfg["threshold"]),
                min_component_pixels=int(
                    cfg["min_component_pixels"]
                ),
                collect_presence=True,
                save_all_predictions=False,
                save_dir=output_dir / "val_preview",
                return_details=True,
            )

        scheduler.step()
        learning_rate = optimizer.param_groups[0]["lr"]

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/patch_val", patch_val_loss, epoch)
        writer.add_scalar("dice/patch_val", patch_mean_dice, epoch)
        writer.add_scalar("lr", learning_rate, epoch)

        metrics: dict[str, Any] = {
            "experiment_name": cfg["experiment_name"],
            "loss_name": "focal_dice",
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "patch_val_loss": patch_val_loss,
            "patch_mean_dice": patch_mean_dice,
            "full_val_mean_dice": (
                None if full_val is None else full_val["mean_dice"]
            ),
            "lr": learning_rate,
        }

        for name, score, count in zip(
            STRUCTURES, patch_class_dice, patch_counts
        ):
            metrics[f"patch_val_{name}_dice"] = score
            metrics[f"patch_val_{name}_count"] = int(count)

        if full_val is not None:
            writer.add_scalar(
                "dice/full_val",
                full_val["mean_dice"],
                epoch,
            )

            for name, score, count in zip(
                STRUCTURES,
                full_val["class_dice"],
                full_val["class_counts"],
            ):
                metrics[f"full_val_{name}_dice"] = score
                metrics[f"full_val_{name}_count"] = int(count)

            target_presence_metrics = presence_metrics_for_targets(
                full_val.get("presence_summary"),
                targets,
            )
            for key, value in target_presence_metrics.items():
                metrics[f"full_val_{key}"] = value

        save_metrics_csv(
            metrics,
            str(output_dir / "metrics.csv"),
        )

        checkpoint_saved = False
        previous_best = best_score

        if (
            full_val is not None
            and full_val["mean_dice"] > best_score
        ):
            best_score = float(full_val["mean_dice"])
            best_epoch = epoch + 1
            checkpoint_saved = True

            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                fold,
                epoch,
                model,
                optimizer,
                scheduler,
                best_score,
            )

            if full_val.get("presence_summary") is not None:
                full_val["presence_summary"].to_csv(
                    output_dir / "best_val_presence_summary.csv",
                    index=False,
                )

        elapsed = time.perf_counter() - epoch_start
        full_val_text = (
            "not evaluated"
            if full_val is None
            else f"{full_val['mean_dice']:.4f}"
        )
        best_text = (
            "not available"
            if best_score == float("-inf")
            else f"{best_score:.4f} (epoch {best_epoch})"
        )

        print("\nEpoch summary", flush=True)
        print(f"  Train loss          : {train_loss:.4f}", flush=True)
        print(f"  Patch val loss      : {patch_val_loss:.4f}", flush=True)
        print(f"  Patch val Dice      : {patch_mean_dice:.4f}", flush=True)
        print(f"  Full-image val Dice : {full_val_text}", flush=True)
        print(f"  Learning rate       : {learning_rate:.3e}", flush=True)
        print(f"  Best full-val Dice  : {best_text}", flush=True)

        if full_val is not None:
            print_target_presence(
                "  Target validation presence:",
                full_val.get("presence_summary"),
                targets,
            )

        if checkpoint_saved:
            print(
                f"  Checkpoint          : SAVED "
                f"({previous_best:.4f} -> {best_score:.4f})",
                flush=True,
            )
        else:
            print("  Checkpoint          : not updated", flush=True)

        print(
            f"  Epoch elapsed       : {format_duration(elapsed)}",
            flush=True,
        )

    writer.close()

    if best_epoch < 0:
        raise RuntimeError(
            f"Fold {fold}: no best checkpoint was saved."
        )

    print("\n" + "=" * 78, flush=True)
    print(
        f"Fold {fold + 1}: training complete. Loading best checkpoint "
        f"from epoch {best_epoch}...",
        flush=True,
    )

    checkpoint = load_best_checkpoint(
        model,
        checkpoint_dir / "best_model.pt",
        device,
    )

    # Independent test is evaluated exactly once after model selection.
    print(
        f"Fold {fold + 1}: running full-image independent TEST once...",
        flush=True,
    )

    test_start = time.perf_counter()

    with torch.no_grad():
        test_results = validate_full_image(
            model=model,
            loader=full_test_loader,
            device=device,
            epoch=int(checkpoint["epoch"]),
            patch_size=int(cfg["patch_size"]),
            stride=int(cfg["stride"]),
            threshold=float(cfg["threshold"]),
            min_component_pixels=int(cfg["min_component_pixels"]),
            collect_presence=True,
            save_all_predictions=bool(
                cfg["save_all_test_predictions"]
            ),
            save_dir=test_prediction_dir,
            return_details=True,
        )

    test_results["presence_per_image"].to_csv(
        output_dir / "test_presence_per_image.csv",
        index=False,
    )

    test_results["presence_summary"].to_csv(
        output_dir / "test_presence_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "structure": STRUCTURES,
            "test_dice": test_results["class_dice"],
            "test_count": test_results["class_counts"],
        }
    ).to_csv(
        output_dir / "test_per_structure_dice.csv",
        index=False,
    )

    test_seconds = time.perf_counter() - test_start
    fold_seconds = time.perf_counter() - fold_start

    print_target_presence(
        "Independent test target presence:",
        test_results.get("presence_summary"),
        targets,
    )

    print(
        f"Fold {fold + 1} test Dice: "
        f"{test_results['mean_dice']:.4f} | "
        f"test elapsed: {format_duration(test_seconds)} | "
        f"fold elapsed: {format_duration(fold_seconds)}",
        flush=True,
    )
    print("=" * 78, flush=True)

    fold_summary = {
        "fold": fold,
        "best_val_score": best_score,
        "best_epoch": best_epoch,
        "test_mean_dice": float(test_results["mean_dice"]),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "n_train_images": int(len(train_idx)),
        "n_val_images": int(len(val_idx)),
        "n_test_images": int(len(test_idx)),
        "test_elapsed": format_duration(test_seconds),
        "fold_elapsed": format_duration(fold_seconds),
    }

    fold_summary.update(
        {
            f"test_{key}": value
            for key, value in presence_metrics_for_targets(
                test_results.get("presence_summary"),
                targets,
            ).items()
        }
    )

    save_json(fold_summary, output_dir / "fold_summary.json")
    complete_marker.write_text(
        (
            f"Fold {fold} completed.\n"
            f"Best epoch: {best_epoch}\n"
            f"Test mean Dice: {test_results['mean_dice']:.6f}\n"
        ),
        encoding="utf-8",
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return fold_summary


def aggregate_results(
    cfg: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    output_root = resolve_repo_path(cfg["output_root"])

    fold_test_scores = [
        float(summary["test_mean_dice"])
        for summary in summaries
    ]

    final_summary = {
        "experiment_name": cfg["experiment_name"],
        "loss_name": "focal_dice",
        "n_splits": int(cfg["n_splits"]),
        "epochs": int(cfg["epochs"]),
        "fold_test_scores": fold_test_scores,
        "mean_test_score": float(
            sum(fold_test_scores) / len(fold_test_scores)
        ),
        "fold_summaries": summaries,
    }

    save_json(final_summary, output_root / "final_summary.json")

    # Aggregate per-structure Dice using valid-instance weighting.
    dice_frames = []
    presence_frames = []

    for fold in range(int(cfg["n_splits"])):
        fold_dir = output_root / f"fold_{fold}"

        dice = pd.read_csv(
            fold_dir / "test_per_structure_dice.csv"
        )
        dice["fold"] = fold
        dice_frames.append(dice)

        presence = pd.read_csv(
            fold_dir / "test_presence_summary.csv"
        )
        presence["fold"] = fold
        presence_frames.append(presence)

    all_dice = pd.concat(dice_frames, ignore_index=True)
    all_presence = pd.concat(presence_frames, ignore_index=True)

    all_dice.to_csv(
        output_root / "all_folds_per_structure_dice.csv",
        index=False,
    )
    all_presence.to_csv(
        output_root / "all_folds_presence_summary.csv",
        index=False,
    )

    weighted_rows = []
    for structure, group in all_dice.groupby("structure", sort=False):
        valid = group["test_count"].fillna(0) > 0
        weights = group.loc[valid, "test_count"].astype(float)
        scores = group.loc[valid, "test_dice"].astype(float)

        weighted_dice = (
            float((scores * weights).sum() / weights.sum())
            if weights.sum() > 0
            else float("nan")
        )

        weighted_rows.append(
            {
                "structure": structure,
                "test_dice": weighted_dice,
                "test_count": float(weights.sum()),
            }
        )

    pd.DataFrame(weighted_rows).to_csv(
        output_root / "test_per_structure_dice_aggregated.csv",
        index=False,
    )

    numeric_presence = [
        column
        for column in (
            "gt_present",
            "pred_present",
            "tp",
            "fp",
            "fn",
            "tn",
            "total_errors",
        )
        if column in all_presence.columns
    ]

    aggregated_presence = (
        all_presence.groupby("structure", as_index=False)[
            numeric_presence
        ].sum()
    )

    aggregated_presence["precision"] = (
        aggregated_presence["tp"]
        / (
            aggregated_presence["tp"]
            + aggregated_presence["fp"]
        ).replace(0, pd.NA)
    )

    aggregated_presence["recall"] = (
        aggregated_presence["tp"]
        / (
            aggregated_presence["tp"]
            + aggregated_presence["fn"]
        ).replace(0, pd.NA)
    )

    aggregated_presence["f1"] = (
        2 * aggregated_presence["tp"]
        / (
            2 * aggregated_presence["tp"]
            + aggregated_presence["fp"]
            + aggregated_presence["fn"]
        ).replace(0, pd.NA)
    )

    aggregated_presence.to_csv(
        output_root / "test_presence_summary_aggregated.csv",
        index=False,
    )

    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/anatomy_aware_dice_focal_5fold.json"
        ),
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional subset of folds, e.g. --folds 0 1. "
            "Default: all five folds."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths and manifests without training.",
    )
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)
    cfg = load_json(config_path)
    seed_everything(int(cfg.get("seed", 42)))

    output_root = resolve_repo_path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output_root / "config_used.json")

    metadata = Metadata(
        str(resolve_repo_path(cfg["metadata_csv"]))
    )
    metadata_df = metadata.dataframe().reset_index(drop=True)

    preflight(cfg, metadata_df)
    if args.preflight_only:
        print("\nPreflight completed successfully.")
        return

    folds = (
        list(range(int(cfg["n_splits"])))
        if args.folds is None or len(args.folds) == 0
        else args.folds
    )

    invalid = [
        fold
        for fold in folds
        if fold < 0 or fold >= int(cfg["n_splits"])
    ]
    if invalid:
        raise ValueError(f"Invalid fold numbers: {invalid}")

    summaries = []
    cv_start = time.perf_counter()

    for fold in folds:
        # Re-seed per fold while preserving reproducibility.
        seed_everything(int(cfg.get("seed", 42)) + fold)
        summary = run_fold(
            fold=fold,
            cfg=cfg,
            metadata_df=metadata_df,
        )
        summaries.append(summary)

        save_json(
            summaries,
            output_root / "cv_summary_partial.json",
        )

    # Full aggregation is only valid when all folds are complete.
    all_summary_paths = [
        output_root / f"fold_{fold}" / "fold_summary.json"
        for fold in range(int(cfg["n_splits"]))
    ]

    if all(path.is_file() for path in all_summary_paths):
        all_summaries = [
            load_json(path)
            for path in all_summary_paths
        ]
        final_summary = aggregate_results(
            cfg,
            all_summaries,
        )

        print("\n" + "=" * 78)
        print("Full anatomy-aware five-fold evaluation finished")
        print(
            f"Fold test Dice: "
            f"{final_summary['fold_test_scores']}"
        )
        print(
            f"Mean test Dice: "
            f"{final_summary['mean_test_score']:.6f}"
        )
        print(
            f"Total elapsed this invocation: "
            f"{format_duration(time.perf_counter() - cv_start)}"
        )
        print("=" * 78)
    else:
        print(
            "\nRequested folds completed. Full aggregation will run "
            "after all five fold summaries are available."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
