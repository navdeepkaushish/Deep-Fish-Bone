#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""One-fold Dice+Focal pilot using the exact Experiment 311 split manifest.

This script deliberately avoids datasets.splitter. It reconstructs fold 0 from:
new_outputs_311/outputs_5fold_311_focal_dice/fold_0/split_manifest.csv

The independent test subset is not evaluated by this pilot. The pilot is for
sampler development using training and validation only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

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


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def normalize_id(value):
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def split_indices_from_manifest(metadata_df, manifest_path):
    manifest = pd.read_csv(manifest_path)
    required = {"split", "image_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"{manifest_path} missing columns: {sorted(missing)}")

    metadata_ids = metadata_df["image_id"].map(normalize_id)
    image_to_index = {
        image_id: index for index, image_id in enumerate(metadata_ids)
    }

    output = {}
    for split in ("train", "val", "test"):
        ids = (
            manifest.loc[
                manifest["split"].astype(str).str.lower() == split,
                "image_id",
            ]
            .map(normalize_id)
            .tolist()
        )
        absent = [image_id for image_id in ids if image_id not in image_to_index]
        if absent:
            raise ValueError(f"{split} IDs absent from metadata: {absent[:10]}")
        output[split] = [image_to_index[image_id] for image_id in ids]

    train_fish = set(metadata_df.iloc[output["train"]]["fish_id"])
    val_fish = set(metadata_df.iloc[output["val"]]["fish_id"])
    test_fish = set(metadata_df.iloc[output["test"]]["fish_id"])
    if not train_fish.isdisjoint(val_fish):
        raise RuntimeError("Fish leakage between train and validation.")
    if not train_fish.isdisjoint(test_fish):
        raise RuntimeError("Fish leakage between train and test.")
    if not val_fish.isdisjoint(test_fish):
        raise RuntimeError("Fish leakage between validation and test.")

    return output["train"], output["val"], output["test"]


def save_checkpoint(path, epoch, model, optimizer, scheduler, score):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "checkpoint_score": score,
        },
        path,
    )


def duration(seconds):
    return str(timedelta(seconds=int(max(0, seconds))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/focal_sampler_pilot_fold0.json",
    )
    args = parser.parse_args()

    cfg = load_json(args.config)
    seed_everything(cfg.get("seed", 42))

    output_dir = Path(cfg["output_root"]) / "fold_0"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output_dir / "config_used.json")

    metadata_df = (
        Metadata(cfg["metadata_csv"])
        .dataframe()
        .reset_index(drop=True)
    )

    train_idx, val_idx, test_idx = split_indices_from_manifest(
        metadata_df=metadata_df,
        manifest_path=Path(cfg["baseline_split_manifest"]),
    )

    print(
        f"Exact baseline split loaded: train={len(train_idx)}, "
        f"val={len(val_idx)}, untouched test={len(test_idx)}"
    )

    train_loader, patch_val_loader = build_patch_loaders(
        root_dir=cfg["root_dir"],
        metadata_csv=cfg["metadata_csv"],
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=cfg["batch_size"],
        patch_size=cfg["patch_size"],
        resize_to=tuple(cfg["resize_to"]),
        num_workers=cfg["num_workers"],
        sampler_config=cfg["sampler"],
    )

    full_val_loader = build_full_image_loader(
        root_dir=cfg["root_dir"],
        metadata_csv=cfg["metadata_csv"],
        indices=val_idx,
        resize_to=tuple(cfg["resize_to"]),
        num_workers=cfg["full_valid_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )
    writer = SummaryWriter(str(output_dir / "tensorboard"))

    best_score = float("-inf")
    best_epoch = -1
    start = time.perf_counter()

    for epoch in range(cfg["epochs"]):
        print(
            f"\nPilot fold 0 | Epoch {epoch + 1}/{cfg['epochs']}",
            flush=True,
        )
        train_loader.dataset.reset_sampler_stats()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        patch_val_loss, patch_mean_dice, patch_class_dice, patch_counts = validate(
            model=model,
            loader=patch_val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
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
            collect_presence=True,
            save_all_predictions=False,
            save_dir=output_dir / "val_preview",
            return_details=True,
        )

        scheduler.step()
        sampler_stats = train_loader.dataset.get_sampler_stats()

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "patch_val_loss": patch_val_loss,
            "patch_mean_dice": patch_mean_dice,
            "full_val_mean_dice": full_val["mean_dice"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        for key, value in sorted(sampler_stats.items()):
            metrics[f"sampler_{key.replace('::', '_')}"] = int(value)

        for name, score, count in zip(
            STRUCTURES,
            full_val["class_dice"],
            full_val["class_counts"],
        ):
            metrics[f"full_val_{name}_dice"] = score
            metrics[f"full_val_{name}_count"] = int(count)

        presence = full_val.get("presence_summary")
        if presence is not None:
            for target in cfg["sampler"]["target_structures"]:
                row = presence.loc[presence["structure"] == target]
                if len(row) == 1:
                    row = row.iloc[0]
                    for key in ("tp", "fp", "fn", "tn"):
                        metrics[f"full_val_{target}_{key}"] = int(row[key])
                    precision = row["tp"] / max(row["tp"] + row["fp"], 1)
                    recall = row["tp"] / max(row["tp"] + row["fn"], 1)
                    f1 = (
                        2 * row["tp"]
                        / max(2 * row["tp"] + row["fp"] + row["fn"], 1)
                    )
                    metrics[f"full_val_{target}_precision"] = precision
                    metrics[f"full_val_{target}_recall"] = recall
                    metrics[f"full_val_{target}_f1"] = f1

        save_metrics_csv(metrics, str(output_dir / "metrics.csv"))
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("dice/full_val", full_val["mean_dice"], epoch)

        if full_val["mean_dice"] > best_score:
            best_score = full_val["mean_dice"]
            best_epoch = epoch + 1
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                best_score,
            )
            if presence is not None:
                presence.to_csv(
                    output_dir / "best_val_presence_summary.csv",
                    index=False,
                )

        print(f"  train loss     : {train_loss:.4f}")
        print(f"  full-val Dice  : {full_val['mean_dice']:.4f}")
        print(f"  best full-val  : {best_score:.4f} (epoch {best_epoch})")
        print(f"  sampler stats  : {sampler_stats}")

    writer.close()

    summary = {
        "best_val_score": best_score,
        "best_epoch": best_epoch,
        "train_images": len(train_idx),
        "val_images": len(val_idx),
        "test_images_untouched": len(test_idx),
        "elapsed": duration(time.perf_counter() - start),
        "test_was_evaluated": False,
    }
    save_json(summary, output_dir / "pilot_summary.json")
    print("\nPilot completed. Independent test set was not evaluated.")
    print(summary)


if __name__ == "__main__":
    main()
