#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Nested grouped CV: patch training, full-image validation, full-image test."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from configs.structure_names import STRUCTURES
from datasets.metadata import Metadata
from datasets.splitter import NestedFishSplitter
from engine.dataloaders import build_full_image_loader, build_patch_loaders
from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from engine.validate_full import validate_full_image
from losses.bce_dice import BCEDiceLoss
from losses.combined import CombinedLoss
from models.unetpp import build_model
from utils.save_metrics import save_metrics_csv
from utils.seed import seed_everything


def load_config(config_path):
    with open(config_path, "r") as handle:
        return json.load(handle)


def save_json(data, path):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=4)


def build_loss(cfg):
    loss_name = cfg.get("loss_name", "focal_dice")
    if loss_name == "focal_dice":
        return CombinedLoss()
    if loss_name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unknown loss_name: {loss_name}")


def save_checkpoint(path, fold, epoch, model, optimizer, scheduler, score):
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


def save_split_manifest(df, fold, train_idx, val_idx, test_idx, output_dir):
    records = []
    for split_name, indices in (
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ):
        subset = df.iloc[indices].copy()
        subset.insert(0, "split", split_name)
        subset.insert(0, "outer_fold", fold)
        records.append(subset)

    manifest = pd.concat(records, ignore_index=True)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    fish_manifest = (
        manifest[["outer_fold", "split", "fish_id", "genotype"]]
        .drop_duplicates()
        .sort_values(["split", "genotype", "fish_id"])
    )
    fish_manifest.to_csv(output_dir / "fish_split_manifest.csv", index=False)


def load_best_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0, seconds))))


def _print_fold_header(fold, cfg, train_idx, val_idx, test_idx, device):
    total_folds = cfg["n_outer_splits"]
    print("\n" + "=" * 78, flush=True)
    print(
        f"OUTER FOLD {fold + 1}/{total_folds} | "
        f"Experiment: {cfg['experiment_name']} | Loss: {cfg['loss_name']}",
        flush=True,
    )
    print(
        f"Images -> train: {len(train_idx)} | val: {len(val_idx)} | "
        f"test: {len(test_idx)} | device: {device}",
        flush=True,
    )
    print("=" * 78, flush=True)


def run_fold(fold, train_idx, val_idx, test_idx, cfg, metadata_df):
    output_root = Path(cfg["output_root"])
    output_dir = output_root / f"fold_{fold}"
    checkpoint_dir = output_dir / "checkpoints"
    test_prediction_dir = output_dir / "test_predictions"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output_dir / "config_used.json")
    save_split_manifest(
        metadata_df, fold, train_idx, val_idx, test_idx, output_dir
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print_fold_header(fold, cfg, train_idx, val_idx, test_idx, device)

    train_loader, patch_val_loader = build_patch_loaders(
        root_dir=cfg["root_dir"],
        metadata_csv=cfg["metadata_csv"],
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=cfg["batch_size"],
        patch_size=cfg["patch_size"],
        resize_to=tuple(cfg["resize_to"]),
        num_workers=cfg["num_workers"],
    )
    full_val_loader = build_full_image_loader(
        root_dir=cfg["root_dir"],
        metadata_csv=cfg["metadata_csv"],
        indices=val_idx,
        resize_to=tuple(cfg["resize_to"]),
        num_workers=cfg["full_valid_workers"],
    )
    full_test_loader = build_full_image_loader(
        root_dir=cfg["root_dir"],
        metadata_csv=cfg["metadata_csv"],
        indices=test_idx,
        resize_to=tuple(cfg["resize_to"]),
        num_workers=cfg["full_test_workers"],
    )

    model = build_model().to(device)
    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    writer = SummaryWriter(str(output_dir / "tensorboard"))

    best_score = float("-inf")
    best_epoch = -1

    fold_start = time.perf_counter()

    for epoch in range(cfg["epochs"]):
        epoch_start = time.perf_counter()
        print("\n" + "-" * 78, flush=True)
        print(
            f"Fold {fold + 1}/{cfg['n_outer_splits']} | "
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
        patch_val_loss, patch_mean_dice, patch_class_dice, patch_counts = validate(
            model=model,
            loader=patch_val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        run_full_val = (
            epoch == 0
            or (epoch + 1) % cfg["full_validation_every"] == 0
        )
        full_val = None
        if run_full_val:
            print(
                "[3/3] Full-image inner-validation "
                "(checkpoint selection)...",
                flush=True,
            )
            full_val = validate_full_image(
                model=model,
                loader=full_val_loader,
                device=device,
                epoch=epoch,
                patch_size=cfg["patch_size"],
                stride=cfg["stride"],
                threshold=cfg["threshold"],
                min_component_pixels=cfg["min_component_pixels"],
                collect_presence=False,
                save_all_predictions=False,
                save_dir=output_dir / "val_preview",
                return_details=True,
            )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/patch_val", patch_val_loss, epoch)
        writer.add_scalar("dice/patch_val", patch_mean_dice, epoch)
        writer.add_scalar("lr", lr, epoch)
        if full_val is not None:
            writer.add_scalar("dice/full_val", full_val["mean_dice"], epoch)

        metrics = {
            "experiment_name": cfg["experiment_name"],
            "loss_name": cfg["loss_name"],
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "patch_val_loss": patch_val_loss,
            "patch_mean_dice": patch_mean_dice,
            "full_val_mean_dice": (
                None if full_val is None else full_val["mean_dice"]
            ),
            "lr": lr,
        }

        for name, score, count in zip(
            STRUCTURES, patch_class_dice, patch_counts
        ):
            metrics[f"patch_val_{name}_dice"] = score
            metrics[f"patch_val_{name}_count"] = int(count)

        if full_val is not None:
            for name, score, count in zip(
                STRUCTURES,
                full_val["class_dice"],
                full_val["class_counts"],
            ):
                metrics[f"full_val_{name}_dice"] = score
                metrics[f"full_val_{name}_count"] = int(count)

        save_metrics_csv(metrics, str(output_dir / "metrics.csv"))

        checkpoint_saved = False
        previous_best = best_score

        # IMPORTANT: checkpoint selection uses only full-image inner-val Dice.
        if full_val is not None and full_val["mean_dice"] > best_score:
            best_score = full_val["mean_dice"]
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

        epoch_seconds = time.perf_counter() - epoch_start
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
        print(f"  Learning rate       : {lr:.3e}", flush=True)
        print(f"  Best full-val Dice  : {best_text}", flush=True)
        if checkpoint_saved:
            print(
                f"  Checkpoint          : SAVED "
                f"({previous_best:.4f} -> {best_score:.4f})",
                flush=True,
            )
        else:
            print("  Checkpoint          : not updated", flush=True)
        print(
            f"  Epoch elapsed       : {_format_duration(epoch_seconds)}",
            flush=True,
        )

    writer.close()

    print("\n" + "=" * 78, flush=True)
    print(
        f"Fold {fold + 1}: training complete. Loading best checkpoint "
        f"from epoch {best_epoch}...",
        flush=True,
    )
    checkpoint = load_best_model(
        model, checkpoint_dir / "best_model.pt", device
    )

    # Outer test is evaluated exactly once, after all model selection is complete.
    print(
        f"Fold {fold + 1}: running full-image OUTER TEST evaluation once...",
        flush=True,
    )
    test_start = time.perf_counter()
    test_results = validate_full_image(
        model=model,
        loader=full_test_loader,
        device=device,
        patch_size=cfg["patch_size"],
        stride=cfg["stride"],
        threshold=cfg["threshold"],
        min_component_pixels=cfg["min_component_pixels"],
        collect_presence=True,
        save_all_predictions=cfg["save_all_test_predictions"],
        save_dir=test_prediction_dir,
        return_details=True,
    )

    test_results["presence_per_image"].to_csv(
        output_dir / "test_presence_per_image.csv", index=False
    )
    test_results["presence_summary"].to_csv(
        output_dir / "test_presence_summary.csv", index=False
    )

    per_structure_test = pd.DataFrame(
        {
            "structure": STRUCTURES,
            "test_dice": test_results["class_dice"],
            "test_count": test_results["class_counts"],
        }
    )
    per_structure_test.to_csv(
        output_dir / "test_per_structure_dice.csv", index=False
    )

    test_seconds = time.perf_counter() - test_start
    fold_seconds = time.perf_counter() - fold_start
    print(
        f"Fold {fold + 1} test Dice: {test_results['mean_dice']:.4f} | "
        f"test elapsed: {_format_duration(test_seconds)} | "
        f"fold elapsed: {_format_duration(fold_seconds)}",
        flush=True,
    )
    print("=" * 78, flush=True)

    fold_summary = {
        "fold": fold,
        "best_val_score": best_score,
        "best_epoch": best_epoch,
        "test_mean_dice": test_results["mean_dice"],
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "n_train_images": int(len(train_idx)),
        "n_val_images": int(len(val_idx)),
        "n_test_images": int(len(test_idx)),
    }
    save_json(fold_summary, output_dir / "fold_summary.json")
    return fold_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/focal_nested_cv.json"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output_root / "config_used.json")

    metadata = Metadata(cfg["metadata_csv"])
    df = metadata.dataframe().reset_index(drop=True)

    splitter = NestedFishSplitter(
        n_outer_splits=cfg["n_outer_splits"],
        inner_val_fraction=cfg["inner_val_fraction"],
        random_state=cfg.get("seed", 42),
    )

    summaries = []
    print("\nStarting nested grouped cross-validation", flush=True)
    print(
        f"Outer folds: {cfg['n_outer_splits']} | Epochs/fold: {cfg['epochs']} | "
        f"Patch size: {cfg['patch_size']} | Stride: {cfg['stride']}",
        flush=True,
    )

    cv_start = time.perf_counter()
    for fold, train_idx, val_idx, test_idx in splitter.split(df):
        summary = run_fold(
            fold, train_idx, val_idx, test_idx, cfg, df
        )
        summaries.append(summary)
        save_json(summaries, output_root / "cv_summary.json")

    final_summary = {
        "experiment_name": cfg["experiment_name"],
        "loss_name": cfg["loss_name"],
        "fold_test_scores": [x["test_mean_dice"] for x in summaries],
        "mean_test_score": float(
            sum(x["test_mean_dice"] for x in summaries) / len(summaries)
        ),
        "fold_summaries": summaries,
    }
    save_json(final_summary, output_root / "final_summary.json")
    total_seconds = time.perf_counter() - cv_start
    print("\n" + "=" * 78, flush=True)
    print("Nested cross-validation finished", flush=True)
    print(
        f"Mean outer-test Dice: {final_summary['mean_test_score']:.4f}",
        flush=True,
    )
    print(
        f"Total elapsed: {_format_duration(total_seconds)}",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(final_summary, flush=True)


if __name__ == "__main__":
    main()
