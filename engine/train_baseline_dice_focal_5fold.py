#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corrected original Dice+Focal five-fold baseline.

Purpose
-------
Retrain the original Dice+Focal baseline from scratch using the FINAL corrected
annotations, while keeping the same fish-level 3/1/1 split manifests and the
same architecture/training/evaluation settings as the later experiments.

Crucially:
- NO anatomy-aware sampler is used.
- Loss is the repository's existing CombinedLoss (Dice + Focal).
- Each fold selects the best checkpoint by full-image validation mean Dice.
- The corresponding independent test fold is evaluated exactly once.
"""

from __future__ import annotations

import argparse
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
from engine.dataloaders import build_patch_loaders, build_full_image_loader
from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from engine.validate_full import validate_full_image
from losses.combined import CombinedLoss
from models.unetpp import build_model
from utils.save_metrics import save_metrics_csv
from utils.seed import seed_everything


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def norm(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def exact_split(df, manifest_path):
    m = pd.read_csv(manifest_path)
    required = {"split", "image_id"}
    missing = required - set(m.columns)
    if missing:
        raise KeyError(
            f"{manifest_path} missing columns {sorted(missing)}; "
            f"available={m.columns.tolist()}"
        )

    metadata_ids = df["image_id"].map(norm)
    if metadata_ids.duplicated().any():
        raise ValueError("Duplicate image IDs in metadata.")
    lookup = {image_id: i for i, image_id in enumerate(metadata_ids)}
    out = {}

    for name in ("train", "val", "test"):
        ids = (
            m.loc[m["split"].astype(str).str.lower() == name, "image_id"]
            .map(norm)
            .tolist()
        )

        missing_ids = [x for x in ids if x not in lookup]
        if missing_ids:
            raise ValueError(
                f"{name} IDs missing from metadata: {missing_ids[:20]}"
            )

        out[name] = [lookup[x] for x in ids]

    # Fish-level leakage check.
    train_fish = set(df.iloc[out["train"]]["fish_id"])
    val_fish = set(df.iloc[out["val"]]["fish_id"])
    test_fish = set(df.iloc[out["test"]]["fish_id"])

    if not train_fish.isdisjoint(val_fish):
        raise RuntimeError("Fish leakage train/val.")
    if not train_fish.isdisjoint(test_fish):
        raise RuntimeError("Fish leakage train/test.")
    if not val_fish.isdisjoint(test_fish):
        raise RuntimeError("Fish leakage val/test.")

    all_indices = out["train"] + out["val"] + out["test"]
    if len(all_indices) != len(set(all_indices)):
        raise RuntimeError("Image leakage between train/val/test subsets.")

    return out["train"], out["val"], out["test"]



EXPECTED_RECORDS = 190
EXPECTED_FISH = 34
EXCLUDED_IMAGE_IDS = {"525410710", "503027953"}
EXPECTED_FOLD_COUNTS = {
    0: (115, 37, 38, 21, 6, 7),
    1: (115, 38, 37, 21, 7, 6),
    2: (113, 39, 38, 20, 7, 7),
    3: (113, 38, 39, 20, 7, 7),
    4: (114, 38, 38, 20, 7, 7),
}

def preflight(cfg, df):
    metadata_path = ROOT / cfg["metadata_csv"]
    manifest_root = ROOT / cfg["manifest_root"]

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if len(df) != EXPECTED_RECORDS:
        raise RuntimeError(f"Expected {EXPECTED_RECORDS} metadata rows, found {len(df)}.")
    if "fish_id" not in df.columns or "image_id" not in df.columns:
        raise KeyError("Metadata must contain image_id and fish_id.")

    ids = df["image_id"].map(norm)
    if ids.duplicated().any():
        raise RuntimeError("Duplicate image_id values found in metadata.")
    if df["fish_id"].nunique() != EXPECTED_FISH:
        raise RuntimeError(
            f"Expected {EXPECTED_FISH} biological fish, found {df['fish_id'].nunique()}."
        )
    bad = sorted(set(ids) & EXCLUDED_IMAGE_IDS)
    if bad:
        raise RuntimeError(f"Excluded image IDs are still present: {bad}")

    all_test_ids = []
    for fold in range(5):
        path = manifest_root / f"fold_{fold}" / "split_manifest.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {path}")
        m = pd.read_csv(path)
        if len(m) != EXPECTED_RECORDS:
            raise RuntimeError(f"Fold {fold}: expected 190 manifest rows, found {len(m)}.")
        if {"split", "image_id"} - set(m.columns):
            raise KeyError(f"Fold {fold}: manifest must contain split and image_id.")

        mids = m["image_id"].map(norm)
        if mids.duplicated().any():
            raise RuntimeError(f"Fold {fold}: duplicate image_id values in manifest.")
        if set(mids) != set(ids):
            raise RuntimeError(f"Fold {fold}: manifest image IDs do not exactly match metadata.")

        # If the frozen manifest stores fish_id, verify it against corrected metadata.
        if "fish_id" in m.columns:
            fish_lookup = dict(zip(ids, df["fish_id"].astype(str)))
            mismatched = [
                image_id for image_id, fish_id in zip(mids, m["fish_id"].astype(str))
                if fish_lookup[image_id] != fish_id
            ]
            if mismatched:
                raise RuntimeError(
                    f"Fold {fold}: manifest fish_id disagrees with metadata for {mismatched[:20]}"
                )

        tr, va, te = exact_split(df, path)
        counts = (
            len(tr), len(va), len(te),
            df.iloc[tr]["fish_id"].nunique(),
            df.iloc[va]["fish_id"].nunique(),
            df.iloc[te]["fish_id"].nunique(),
        )
        if counts != EXPECTED_FOLD_COUNTS[fold]:
            raise RuntimeError(
                f"Fold {fold}: counts {counts}, expected {EXPECTED_FOLD_COUNTS[fold]}."
            )
        all_test_ids.extend(df.iloc[te]["image_id"].map(norm).tolist())
        print(
            f"Fold {fold}: train={counts[0]}/{counts[3]} fish, "
            f"val={counts[1]}/{counts[4]} fish, test={counts[2]}/{counts[5]} fish"
        )

    if len(all_test_ids) != EXPECTED_RECORDS or len(set(all_test_ids)) != EXPECTED_RECORDS:
        raise RuntimeError("Each of the 190 records must appear exactly once as test across folds.")

    print("Preflight PASS: frozen 190 records, 34 fish, five leakage-free rotations.")

def build_baseline_patch_loaders(cfg, train_idx, val_idx):
    """
    Deliberately disables the anatomy-aware sampler.

    The current repository dataloader accepts sampler_config. Passing None
    restores the ordinary/original FishDataset patch sampling behavior.
    """
    return build_patch_loaders(
        root_dir=str(ROOT / cfg["root_dir"]),
        metadata_csv=str(ROOT / cfg["metadata_csv"]),
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=cfg["batch_size"],
        patch_size=cfg["patch_size"],
        resize_to=tuple(cfg["resize_to"]),
        num_workers=cfg["num_workers"],
        sampler_config=None,
    )


def train_fold(fold, cfg, df, device):
    fold_dir = ROOT / cfg["output_root"] / f"fold_{fold}"
    ckdir = fold_dir / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)

    marker = fold_dir / "FOLD_COMPLETE.marker"

    if marker.exists() and not cfg.get("rerun_completed_folds", False):
        print(f"Fold {fold} already complete; skipping.")
        return

    manifest = (
        ROOT
        / cfg["manifest_root"]
        / f"fold_{fold}"
        / "split_manifest.csv"
    )

    tr, va, te = exact_split(df, manifest)

    fold_seed = int(cfg["seed"]) + fold
    seed_everything(fold_seed)

    fold_cfg = json.loads(json.dumps(cfg))
    fold_cfg["fold"] = fold
    fold_cfg["seed"] = fold_seed
    save_json(fold_cfg, fold_dir / "config_used.json")

    print("\n" + "=" * 80)
    print(
        f"Fold {fold+1}/5 | train={len(tr)} "
        f"val={len(va)} test={len(te)}"
    )
    print("Loss: repository CombinedLoss = Dice + Focal")
    print("Sampler: ORIGINAL / standard patch sampling (anatomy sampler OFF)")
    print("=" * 80)

    train_loader, patch_val_loader = build_baseline_patch_loaders(
        cfg, tr, va
    )

    full_val_loader = build_full_image_loader(
        root_dir=str(ROOT / cfg["root_dir"]),
        metadata_csv=str(ROOT / cfg["metadata_csv"]),
        indices=va,
        resize_to=tuple(cfg["resize_to"]),
        num_workers=cfg["full_valid_workers"],
    )

    model = build_model().to(device)
    criterion = CombinedLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best = -1.0
    best_epoch = -1

    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        patch_val_loss, patch_mean_dice, _, _ = validate(
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
            save_dir=fold_dir / "val_preview",
            return_details=True,
        )

        scheduler.step()

        row = {
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "patch_val_loss": patch_val_loss,
            "patch_mean_dice": patch_mean_dice,
            "full_val_mean_dice": full_val["mean_dice"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        for s, d, c in zip(
            STRUCTURES,
            full_val["class_dice"],
            full_val["class_counts"],
        ):
            row[f"full_val_{s}_dice"] = d
            row[f"full_val_{s}_count"] = int(c)

        save_metrics_csv(
            row,
            str(fold_dir / "metrics.csv"),
        )

        if full_val["mean_dice"] > best:
            best = float(full_val["mean_dice"])
            best_epoch = epoch + 1

            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "checkpoint_score": best,
                },
                ckdir / "best_model.pt",
            )

            pd.DataFrame(
                {
                    "structure": STRUCTURES,
                    "val_dice": full_val["class_dice"],
                    "val_count": full_val["class_counts"],
                }
            ).to_csv(
                fold_dir / "best_val_per_structure_dice.csv",
                index=False,
            )

            if full_val.get("presence_summary") is not None:
                full_val["presence_summary"].to_csv(
                    fold_dir / "best_val_presence_summary.csv",
                    index=False,
                )

        print(
            f"Fold {fold+1}/5 Epoch {epoch+1}/{cfg['epochs']} | "
            f"train={train_loss:.4f} "
            f"patch_val={patch_val_loss:.4f} "
            f"full_val={full_val['mean_dice']:.4f} "
            f"best={best:.4f}@{best_epoch}"
        )

    # ----------------------------------------------------------
    # Independent test evaluation once, after checkpoint selection.
    # ----------------------------------------------------------
    ck = torch.load(
        ckdir / "best_model.pt",
        map_location=device,
    )

    model.load_state_dict(
        ck["model_state_dict"],
        strict=True,
    )
    model.eval()

    test_loader = build_full_image_loader(
        root_dir=str(ROOT / cfg["root_dir"]),
        metadata_csv=str(ROOT / cfg["metadata_csv"]),
        indices=te,
        resize_to=tuple(cfg["resize_to"]),
        num_workers=cfg["full_test_workers"],
    )

    test_dir = fold_dir / "independent_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        r = validate_full_image(
            model=model,
            loader=test_loader,
            device=device,
            epoch=int(ck["epoch"]),
            patch_size=cfg["patch_size"],
            stride=cfg["stride"],
            threshold=cfg["threshold"],
            min_component_pixels=cfg["min_component_pixels"],
            collect_presence=True,
            save_all_predictions=True,
            save_dir=test_dir / "predictions",
            return_details=True,
        )

    pd.DataFrame(
        {
            "structure": STRUCTURES,
            "test_dice": r["class_dice"],
            "test_count": r["class_counts"],
        }
    ).to_csv(
        test_dir / "test_per_structure_dice.csv",
        index=False,
    )

    r["presence_summary"].to_csv(
        test_dir / "test_presence_summary.csv",
        index=False,
    )

    r["presence_per_image"].to_csv(
        test_dir / "test_presence_per_image.csv",
        index=False,
    )

    summary = {
        "fold": fold,
        "best_epoch": int(ck["epoch"]) + 1,
        "best_validation_dice": float(ck["checkpoint_score"]),
        "test_mean_dice": float(r["mean_dice"]),
        "n_train_images": len(tr),
        "n_val_images": len(va),
        "n_test_images": len(te),
        "threshold": cfg["threshold"],
        "min_component_pixels": cfg["min_component_pixels"],
        "anatomy_sampler": False,
        "loss": "Dice+Focal",
    }

    save_json(
        summary,
        test_dir / "test_summary.json",
    )

    print("\nIndependent test result")
    print(json.dumps(summary, indent=2))

    focus = r["presence_summary"][
        r["presence_summary"]["structure"].isin(
            ["Br2a", "Br2b"]
        )
    ].copy()

    dice_df = pd.DataFrame(
        {
            "structure": STRUCTURES,
            "test_dice": r["class_dice"],
        }
    )

    focus = focus.merge(
        dice_df,
        on="structure",
    )

    print("\nBr2 results")
    for _, x in focus.iterrows():
        tp = int(x.tp)
        fp = int(x.fp)
        fn = int(x.fn)

        print(
            f"{x.structure}: Dice={x.test_dice:.4f} "
            f"TP={tp} FP={fp} FN={fn} "
            f"P={tp/max(tp+fp,1):.3f} "
            f"R={tp/max(tp+fn,1):.3f}"
        )

    marker.write_text(
        "complete\n",
        encoding="utf-8",
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def aggregate(cfg):
    root = ROOT / cfg["output_root"]

    summaries = []
    dice_frames = []
    presence_frames = []

    for fold in range(5):
        test_dir = (
            root
            / f"fold_{fold}"
            / "independent_test"
        )

        with open(
            test_dir / "test_summary.json",
            "r",
            encoding="utf-8",
        ) as f:
            summaries.append(json.load(f))

        d = pd.read_csv(
            test_dir / "test_per_structure_dice.csv"
        )
        d["fold"] = fold
        dice_frames.append(d)

        p = pd.read_csv(
            test_dir / "test_presence_summary.csv"
        )
        p["fold"] = fold
        presence_frames.append(p)

    fold_df = pd.DataFrame(
        summaries
    ).sort_values("fold")

    fold_df.to_csv(
        root / "fold_test_summary.csv",
        index=False,
    )

    all_dice = pd.concat(
        dice_frames,
        ignore_index=True,
    )

    all_presence = pd.concat(
        presence_frames,
        ignore_index=True,
    )

    all_dice.to_csv(
        root / "all_folds_per_structure_dice.csv",
        index=False,
    )

    all_presence.to_csv(
        root / "all_folds_presence_summary.csv",
        index=False,
    )

    # Weighted per-structure Dice across independent test folds.
    pooled_dice = []

    for structure, g in all_dice.groupby(
        "structure",
        sort=False,
    ):
        n = g["test_count"].sum()

        weighted = (
            (g["test_dice"] * g["test_count"]).sum() / n
            if n > 0
            else float("nan")
        )

        pooled_dice.append(
            {
                "structure": structure,
                "weighted_test_dice": weighted,
                "test_count": int(n),
            }
        )

    pd.DataFrame(
        pooled_dice
    ).to_csv(
        root / "pooled_per_structure_dice.csv",
        index=False,
    )

    # Pooled presence counts.
    pooled_presence = []

    for structure, g in all_presence.groupby(
        "structure",
        sort=False,
    ):
        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        fn = int(g["fn"].sum())
        tn = int(g["tn"].sum())

        pooled_presence.append(
            {
                "structure": structure,
                "gt_present": tp + fn,
                "pred_present": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "total_errors": fp + fn,
                "precision": (
                    tp / (tp + fp)
                    if tp + fp
                    else float("nan")
                ),
                "recall": (
                    tp / (tp + fn)
                    if tp + fn
                    else float("nan")
                ),
                "f1": (
                    2 * tp / (2 * tp + fp + fn)
                    if 2 * tp + fp + fn
                    else float("nan")
                ),
                "specificity": (
                    tn / (tn + fp)
                    if tn + fp
                    else float("nan")
                ),
            }
        )

    pooled_presence = pd.DataFrame(
        pooled_presence
    )

    pooled_presence.to_csv(
        root / "pooled_presence_summary.csv",
        index=False,
    )

    final = {
        "mean_fold_test_dice": float(
            fold_df["test_mean_dice"].mean()
        ),
        "sample_sd_fold_test_dice": float(
            fold_df["test_mean_dice"].std(ddof=1)
        ),
        "fold_scores": fold_df[
            "test_mean_dice"
        ].tolist(),
        "loss": "Dice+Focal",
        "anatomy_sampler": False,
    }

    save_json(
        final,
        root / "final_summary.json",
    )

    print("\n" + "=" * 80)
    print("FINAL CORRECTED ORIGINAL DICE+FOCAL BASELINE")
    print("=" * 80)

    print(
        fold_df[
            [
                "fold",
                "best_epoch",
                "best_validation_dice",
                "test_mean_dice",
            ]
        ].to_string(index=False)
    )

    print(
        "\nMean test Dice = "
        f"{final['mean_fold_test_dice']:.6f} "
        f"± {final['sample_sd_fold_test_dice']:.6f}"
    )

    print("\nPooled Br2")
    print(
        pooled_presence[
            pooled_presence["structure"].isin(
                ["Br2a", "Br2b"]
            )
        ].to_string(index=False)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default=(
            "configs/"
            "baseline_dice_focal_5fold.json"
        ),
    )

    ap.add_argument(
        "--folds",
        nargs="*",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--aggregate-only",
        action="store_true",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen metadata and manifests without training.",
    )

    args = ap.parse_args()

    cfg = load_json(
        ROOT / args.config
    )

    df = Metadata(
        str(ROOT / cfg["metadata_csv"])
    ).dataframe().reset_index(drop=True)

    preflight(cfg, df)
    if args.preflight_only:
        return

    if args.aggregate_only:
        aggregate(cfg)
        return

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    folds = (
        list(range(5))
        if not args.folds
        else args.folds
    )

    for fold in folds:
        train_fold(
            fold,
            cfg,
            df,
            device,
        )

    if all(
        (
            ROOT
            / cfg["output_root"]
            / f"fold_{fold}"
            / "FOLD_COMPLETE.marker"
        ).exists()
        for fold in range(5)
    ):
        aggregate(cfg)


if __name__ == "__main__":
    main()
