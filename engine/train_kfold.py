#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.seed import seed_everything
from utils.save_metrics import save_metrics_csv

from datasets.metadata import Metadata
from datasets.splitter import FishSplitter

from engine.dataloaders import (
    build_dataloaders,
    build_full_valid_loader
)

from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from engine.validate_full import validate_full_image

from models.unetpp import build_model
from losses.combined import CombinedLoss

from configs.structure_names import STRUCTURES


ROOT_DIR = "ventral"
METADATA_CSV = "metadata/metadata.csv"


def run_fold(
    fold,
    train_idx,
    valid_idx,
    epochs=100,
    batch_size=2,
    num_workers=6,
    full_valid_workers=4,
    lr=3e-4,
    output_root="outputs_cv_focal"
):

    output_dir = Path(output_root) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 70)
    print(f"Starting Fold {fold}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print("=" * 70)

    train_loader, valid_loader = build_dataloaders(
        root_dir=ROOT_DIR,
        metadata_csv=METADATA_CSV,
        train_idx=train_idx,
        valid_idx=valid_idx,
        batch_size=batch_size,
        patch_size=512,
        resize_to=(966, 1288),
        num_workers=num_workers
    )

    full_valid_loader = build_full_valid_loader(
        root_dir=ROOT_DIR,
        metadata_csv=METADATA_CSV,
        valid_idx=valid_idx,
        resize_to=(966, 1288),
        num_workers=full_valid_workers
    )

    model = build_model().to(device)

    criterion = CombinedLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    writer = SummaryWriter(
        str(output_dir / "tensorboard")
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_score = 0.0

    for epoch in range(epochs):

        print("\n" + "=" * 60)
        print(f"Fold {fold} | Epoch {epoch + 1}/{epochs}")
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

        if epoch == 0 or (epoch + 1) % 5 == 0:

            (
                full_mean_dice,
                full_class_dice,
                full_class_counts
            ) = validate_full_image(
                model=model,
                loader=full_valid_loader,
                device=device,
                epoch=epoch,
                patch_size=512,
                stride=256,
                threshold=0.5
            )

        else:

            full_mean_dice = None
            full_class_dice = None
            full_class_counts = None

        scheduler.step()

        print(f"Train Loss:      {train_loss:.4f}")
        print(f"Patch Val Loss:  {patch_val_loss:.4f}")
        print(f"Patch Dice:      {patch_mean_dice:.4f}")

        if full_mean_dice is not None:
            print(f"Full Dice:       {full_mean_dice:.4f}")
        else:
            print("Full Dice:       not evaluated this epoch")

        print(f"LR:              {optimizer.param_groups[0]['lr']:.8f}")

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU Memory: {peak_mem:.2f} GB")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/patch_val", patch_val_loss, epoch)
        writer.add_scalar("dice/patch_val", patch_mean_dice, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if full_mean_dice is not None:
            writer.add_scalar("dice/full_val", full_mean_dice, epoch)

        metrics = {
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "patch_val_loss": patch_val_loss,
            "patch_mean_dice": patch_mean_dice,
            "full_mean_dice": full_mean_dice,
            "lr": optimizer.param_groups[0]["lr"]
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
                checkpoint_dir / "best_model.pt"
            )

            print(f"Saved best model: {checkpoint_score:.4f}")

        if (epoch + 1) % 5 == 0:

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
                checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            )

    writer.close()

    return best_score


def main():

    seed_everything(42)

    meta = Metadata(METADATA_CSV)
    df = meta.dataframe()

    splitter = FishSplitter(n_splits=5)

    all_scores = []

    for fold, train_idx, valid_idx in splitter.split(df):

        score = run_fold(
            fold=fold,
            train_idx=train_idx,
            valid_idx=valid_idx,
            epochs=100,
            batch_size=2,
            num_workers=6,
            full_valid_workers=4,
            lr=3e-4,
            output_root="outputs_cv_focal"
        )

        all_scores.append(score)

    print("\nCross-validation finished")
    print("Fold scores:", all_scores)
    print("Mean score:", sum(all_scores) / len(all_scores))


if __name__ == "__main__":

    main()