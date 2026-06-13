#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:19:52 2026

@author: navdeepkaushish
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.seed import seed_everything

from datasets.metadata import Metadata
from datasets.splitter import FishSplitter

from engine.dataloaders import (
    build_dataloaders,
    build_full_valid_loader
)

from models.unetpp import build_model
from losses.combined import CombinedLoss

from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from engine.validate_full import validate_full_image

from configs.structure_names import STRUCTURES
from utils.save_metrics import save_metrics_csv


def main():

    seed_everything(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ------------------------------
    # Metadata
    # ------------------------------

    meta = Metadata("metadata/metadata.csv")
    df = meta.dataframe()

    # ------------------------------
    # Split
    # ------------------------------

    splitter = FishSplitter(n_splits=5)

    fold, train_idx, valid_idx = next(
        splitter.split(df)
    )

    print(f"Running Fold {fold}")

    # ------------------------------
    # Dataloaders
    # ------------------------------

    train_loader, valid_loader = build_dataloaders(
        root_dir="ventral",
        metadata_csv="metadata/metadata.csv",
        train_idx=train_idx,
        valid_idx=valid_idx,
        batch_size=2,
        patch_size=512,
        resize_to=(966, 1288),
        num_workers=0
    )

    full_valid_loader = build_full_valid_loader(
        root_dir="ventral",
        metadata_csv="metadata/metadata.csv",
        valid_idx=valid_idx,
        resize_to=(966, 1288),
        num_workers=0
    )

    # ------------------------------
    # Model
    # ------------------------------

    model = build_model()
    model = model.to(device)

    # ------------------------------
    # Loss / Optimizer / Scheduler
    # ------------------------------

    criterion = CombinedLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    epochs = 10

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    # ------------------------------
    # Logging / Outputs
    # ------------------------------

    writer = SummaryWriter("outputs/tensorboard")

    Path("outputs/checkpoints").mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Structures: {len(STRUCTURES)}")

    best_dice = 0.0

    # ==============================
    # Training Loop
    # ==============================

    for epoch in range(epochs):

        print("\n" + "=" * 50)
        print(f"Epoch {epoch + 1}/{epochs}")
        print("=" * 50)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # --------------------------
        # Train on random patches
        # --------------------------

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )

        # --------------------------
        # Fast patch validation
        # --------------------------

        (
            val_loss,
            val_dice,
            class_dice,
            class_counts
        ) = validate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        # --------------------------
        # Full-image validation
        # --------------------------

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

        # --------------------------
        # Scheduler
        # --------------------------

        scheduler.step()

        # --------------------------
        # Console logging
        # --------------------------

        print()
        print(f"Train Loss:       {train_loss:.4f}")
        print(f"Patch Val Loss:   {val_loss:.4f}")
        print(f"Patch Mean Dice:  {val_dice:.4f}")

        if full_mean_dice is not None:
            print(f"Full Mean Dice:   {full_mean_dice:.4f}")
        else:
            print("Full Mean Dice:   not evaluated this epoch")

        print(f"LR:               {optimizer.param_groups[0]['lr']:.8f}")
        print(f"Valid Structures: {sum(c > 0 for c in class_counts)}/25")

        if device.type == "cuda":

            peak_mem = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            print(f"Peak GPU Memory:  {peak_mem:.2f} GB")

        # --------------------------
        # Per-structure patch Dice
        # --------------------------

        print("\nPatch Per-Structure Dice")
        print("-" * 45)

        for name, score, count in zip(
            STRUCTURES,
            class_dice,
            class_counts
        ):

            if score is None:
                print(f"{name:<5s} NA (n=0)")
            else:
                print(f"{name:<5s} {score:.4f} (n={int(count)})")

        # --------------------------
        # Per-structure full Dice
        # --------------------------

        if full_class_dice is not None:

            print("\nFull-Image Per-Structure Dice")
            print("-" * 45)

            for name, score, count in zip(
                STRUCTURES,
                full_class_dice,
                full_class_counts
            ):

                if score is None:
                    print(f"{name:<5s} NA (n=0)")
                else:
                    print(f"{name:<5s} {score:.4f} (n={int(count)})")

        # --------------------------
        # TensorBoard
        # --------------------------

        writer.add_scalar(
            "loss/train",
            train_loss,
            epoch
        )

        writer.add_scalar(
            "loss/patch_val",
            val_loss,
            epoch
        )

        writer.add_scalar(
            "dice/patch_val",
            val_dice,
            epoch
        )

        if full_mean_dice is not None:
            writer.add_scalar(
                "dice/full_image_val",
                full_mean_dice,
                epoch
            )

        writer.add_scalar(
            "lr",
            optimizer.param_groups[0]["lr"],
            epoch
        )

        # --------------------------
        # Save metrics CSV
        # --------------------------

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "patch_val_loss": val_loss,
            "patch_mean_dice": val_dice,
            "full_mean_dice": full_mean_dice,
            "lr": optimizer.param_groups[0]["lr"]
        }

        for name, score, count in zip(
            STRUCTURES,
            class_dice,
            class_counts
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
            "outputs/metrics.csv"
        )

        # --------------------------
        # Save checkpoint
        # --------------------------

        checkpoint_score = (
            full_mean_dice
            if full_mean_dice is not None
            else val_dice
        )

        if checkpoint_score > best_dice:

            best_dice = checkpoint_score

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "checkpoint_score": checkpoint_score,
                    "patch_val_dice": val_dice,
                    "full_val_dice": full_mean_dice
                },
                "outputs/checkpoints/best_model.pt"
            )

            print(
                f"\nSaved Best Model "
                f"(score={checkpoint_score:.4f})"
            )

    writer.close()

    print("\nTraining Finished")
    print(f"Best Dice/Score: {best_dice:.4f}")


if __name__ == "__main__":

    main()