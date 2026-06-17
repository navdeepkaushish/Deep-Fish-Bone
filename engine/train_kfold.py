#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from configs.structure_names import STRUCTURES

from datasets.metadata import Metadata
from datasets.splitter import FishSplitter

from engine.dataloaders import (
    build_dataloaders,
    build_full_valid_loader
)

from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from engine.validate_full import validate_full_image

from losses.combined import CombinedLoss
from losses.bce_dice import BCEDiceLoss

from models.unetpp import build_model

from utils.save_metrics import save_metrics_csv
from utils.seed import seed_everything


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def build_loss(cfg):
    loss_name = cfg.get("loss_name", "focal_dice")

    if loss_name == "focal_dice":
        return CombinedLoss()

    if loss_name == "bce_dice":
        return BCEDiceLoss()

    raise ValueError(f"Unknown loss_name: {loss_name}")


def save_checkpoint(
    path,
    fold,
    epoch,
    model,
    optimizer,
    scheduler,
    checkpoint_score,
    patch_mean_dice,
    full_mean_dice
):
    torch.save(
        {
            "fold": fold,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "checkpoint_score": checkpoint_score,
            "patch_mean_dice": patch_mean_dice,
            "full_mean_dice": full_mean_dice
        },
        path
    )


def archive_fold(output_dir):
    output_dir = Path(output_dir)

    archive_path = shutil.make_archive(
        base_name=str(output_dir),
        format="gztar",
        root_dir=output_dir.parent,
        base_dir=output_dir.name
    )

    print(f"Archived fold: {archive_path}")

    return archive_path


def fold_is_complete(output_dir):
    output_dir = Path(output_dir)

    required_files = [
        output_dir / "fold_summary.json",
        output_dir / "metrics.csv",
        output_dir / "checkpoints" / "best_model.pt"
    ]

    return all(path.exists() for path in required_files)


def load_fold_summary(output_dir):
    with open(Path(output_dir) / "fold_summary.json", "r") as f:
        return json.load(f)


def run_fold(
    fold,
    train_idx,
    valid_idx,
    cfg
):
    output_root = Path(cfg["output_root"])
    output_dir = output_root / f"fold_{fold}"
    checkpoint_dir = output_dir / "checkpoints"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_json(cfg, output_dir / "config_used.json")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 70)
    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Loss: {cfg.get('loss_name', 'unknown')}")
    print(f"Fold: {fold}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print("=" * 70)

    train_loader, valid_loader = build_dataloaders(
        root_dir=cfg["root_dir"],
        metadata_csv=cfg["metadata_csv"],
        train_idx=train_idx,
        valid_idx=valid_idx,
        batch_size=cfg["batch_size"],
        patch_size=cfg["patch_size"],
        resize_to=tuple(cfg["resize_to"]),
        num_workers=cfg["num_workers"]
    )

    full_valid_loader = build_full_valid_loader(
        root_dir=cfg["root_dir"],
        metadata_csv=cfg["metadata_csv"],
        valid_idx=valid_idx,
        resize_to=tuple(cfg["resize_to"]),
        num_workers=cfg["full_valid_workers"]
    )

    model = build_model().to(device)

    criterion = build_loss(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    writer = SummaryWriter(str(output_dir / "tensorboard"))

    best_score = 0.0
    best_epoch = -1

    for epoch in range(cfg["epochs"]):

        print("\n" + "=" * 60)
        print(f"Fold {fold} | Epoch {epoch + 1}/{cfg['epochs']}")
        print("=" * 60)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )

        (
            patch_val_loss,
            patch_mean_dice,
            patch_class_dice,
            patch_class_counts
        ) = validate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        should_run_full_validation = (
            epoch == 0
            or (epoch + 1) % cfg["full_validation_every"] == 0
        )

        if should_run_full_validation:
            (
                full_mean_dice,
                full_class_dice,
                full_class_counts
            ) = validate_full_image(
                model=model,
                loader=full_valid_loader,
                device=device,
                epoch=epoch,
                patch_size=cfg["patch_size"],
                stride=cfg["stride"],
                threshold=cfg["threshold"]
            )
        else:
            full_mean_dice = None
            full_class_dice = None
            full_class_counts = None

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print()
        print(f"Train Loss:      {train_loss:.4f}")
        print(f"Patch Val Loss:  {patch_val_loss:.4f}")
        print(f"Patch Dice:      {patch_mean_dice:.4f}")

        if full_mean_dice is not None:
            print(f"Full Dice:       {full_mean_dice:.4f}")
        else:
            print("Full Dice:       not evaluated")

        print(f"LR:              {lr:.8f}")

        valid_structures = sum(c > 0 for c in patch_class_counts)

        print(f"Valid Structures: {valid_structures}/25")

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU Memory: {peak_mem:.2f} GB")
        else:
            peak_mem = None

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/patch_val", patch_val_loss, epoch)
        writer.add_scalar("dice/patch_val", patch_mean_dice, epoch)
        writer.add_scalar("lr", lr, epoch)

        if full_mean_dice is not None:
            writer.add_scalar("dice/full_val", full_mean_dice, epoch)

        metrics = {
            "experiment_name": cfg["experiment_name"],
            "loss_name": cfg.get("loss_name", "unknown"),
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "patch_val_loss": patch_val_loss,
            "patch_mean_dice": patch_mean_dice,
            "full_mean_dice": full_mean_dice,
            "lr": lr,
            "peak_gpu_memory_gb": peak_mem
        }

        for name, score, count in zip(
            STRUCTURES,
            patch_class_dice,
            patch_class_counts
        ):
            metrics[f"patch_{name}_dice"] = score
            metrics[f"patch_{name}_count"] = int(count)

        if full_class_dice is not None:
            for name, score, count in zip(
                STRUCTURES,
                full_class_dice,
                full_class_counts
            ):
                metrics[f"full_{name}_dice"] = score
                metrics[f"full_{name}_count"] = int(count)

        save_metrics_csv(
            metrics,
            str(output_dir / "metrics.csv")
        )

        checkpoint_score = (
            full_mean_dice
            if full_mean_dice is not None
            else patch_mean_dice
        )

        if checkpoint_score > best_score:

            best_score = checkpoint_score
            best_epoch = epoch + 1

            print(
                f"\n✓ Fold {fold} | New Best Full Dice = {best_score:.4f} "
                f"(Epoch {epoch + 1}) --> Saving best_model.pt"
            )

            save_checkpoint(
                path=checkpoint_dir / "best_model.pt",
                fold=fold,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_score=checkpoint_score,
                patch_mean_dice=patch_mean_dice,
                full_mean_dice=full_mean_dice
            )

        checkpoint_every = cfg.get("checkpoint_every", 0)

        if (
            checkpoint_every > 0
            and (epoch + 1) % checkpoint_every == 0
        ):
            save_checkpoint(
                path=checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
                fold=fold,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_score=checkpoint_score,
                patch_mean_dice=patch_mean_dice,
                full_mean_dice=full_mean_dice
            )

    writer.close()

    fold_summary = {
        "fold": fold,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "output_dir": str(output_dir)
    }

    save_json(
        fold_summary,
        output_dir / "fold_summary.json"
    )

    if cfg.get("archive_each_fold", False):
        archive_fold(output_dir)

    return fold_summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/focal_cv.json"
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    seed_everything(42)

    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    save_json(
        cfg,
        output_root / "config_used.json"
    )

    meta = Metadata(cfg["metadata_csv"])
    df = meta.dataframe()

    splitter = FishSplitter(
        n_splits=cfg["n_splits"]
    )

    summaries = []

    for fold, train_idx, valid_idx in splitter.split(df):

        output_dir = output_root / f"fold_{fold}"

        if fold_is_complete(output_dir):

            print(
                f"\n✓ Fold {fold} already complete. Skipping."
            )

            summary = load_fold_summary(output_dir)
            summaries.append(summary)

            save_json(
                summaries,
                output_root / "cv_summary.json"
            )

            continue

        summary = run_fold(
            fold=fold,
            train_idx=train_idx,
            valid_idx=valid_idx,
            cfg=cfg
        )

        summaries.append(summary)

        save_json(
            summaries,
            output_root / "cv_summary.json"
        )

    scores = [
        item["best_score"]
        for item in summaries
    ]

    final_summary = {
        "experiment_name": cfg["experiment_name"],
        "loss_name": cfg.get("loss_name", "unknown"),
        "fold_scores": scores,
        "mean_score": sum(scores) / len(scores),
        "fold_summaries": summaries
    }

    save_json(
        final_summary,
        output_root / "final_summary.json"
    )

    print("\nCross-validation finished")
    print(final_summary)


if __name__ == "__main__":
    main()