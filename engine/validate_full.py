#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Full-image sliding-window evaluation and missing-structure statistics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from configs.structure_names import STRUCTURES
from inference.sliding_window import sliding_window_predict
from metrics.per_structure_dice import per_structure_dice
from visualization.save_prediction import save_prediction


def _remove_small_components(binary_mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove connected foreground components smaller than min_pixels."""
    binary_mask = binary_mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    cleaned = np.zeros_like(binary_mask)
    for label in range(1, n_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_pixels:
            cleaned[labels == label] = 1
    return cleaned


def _presence_rows(
    image_id,
    gt_mask: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float,
    min_component_pixels: int,
):
    gt_np = (gt_mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    pred_np = (
        probabilities.detach().cpu().numpy() >= threshold
    ).astype(np.uint8)

    rows = []
    cleaned_predictions = np.zeros_like(pred_np)

    for class_idx, structure in enumerate(STRUCTURES):
        cleaned = _remove_small_components(
            pred_np[class_idx], min_component_pixels
        )
        cleaned_predictions[class_idx] = cleaned

        gt_present = bool(gt_np[class_idx].any())
        pred_present = bool(cleaned.any())

        rows.append(
            {
                "image_id": str(image_id),
                "structure": structure,
                "gt_present": int(gt_present),
                "pred_present": int(pred_present),
                "tp": int(gt_present and pred_present),
                "fp": int((not gt_present) and pred_present),
                "fn": int(gt_present and (not pred_present)),
                "tn": int((not gt_present) and (not pred_present)),
            }
        )

    return rows, torch.from_numpy(cleaned_predictions).float()


def validate_full_image(
    model,
    loader,
    device,
    epoch=None,
    patch_size=512,
    stride=256,
    threshold=0.5,
    min_component_pixels=25,
    collect_presence=False,
    save_all_predictions=False,
    save_dir=None,
    return_details=False,
):
    """
    Evaluate complete images using sliding-window inference.

    Dice is calculated using the existing per_structure_dice implementation.
    Presence/absence metrics use thresholded predictions after removal of
    connected components smaller than min_component_pixels.
    """
    model.eval()
    n_classes = len(STRUCTURES)

    class_dice_sum = np.zeros(n_classes, dtype=np.float64)
    class_valid_count = np.zeros(n_classes, dtype=np.float64)
    presence_records = []

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            image = batch["image"][0]
            mask = batch["mask"][0].to(device)
            image_id = str(batch["image_id"][0])

            probs = sliding_window_predict(
                model=model,
                image=image,
                device=device,
                patch_size=patch_size,
                stride=stride,
            )

            logits = torch.logit(
                probs.clamp(1e-6, 1.0 - 1e-6)
            ).unsqueeze(0)

            dice_sum, counts = per_structure_dice(
                logits,
                mask.unsqueeze(0),
                threshold=threshold,
            )
            class_dice_sum += np.asarray(dice_sum, dtype=float)
            class_valid_count += np.asarray(counts, dtype=float)

            cleaned_preds = None
            if collect_presence or save_all_predictions:
                rows, cleaned_preds = _presence_rows(
                    image_id=image_id,
                    gt_mask=mask,
                    probabilities=probs,
                    threshold=threshold,
                    min_component_pixels=min_component_pixels,
                )
                if collect_presence:
                    presence_records.extend(rows)

            should_save = (
                save_dir is not None
                and (
                    save_all_predictions
                    or (batch_idx == 0 and epoch is not None)
                )
            )
            if should_save:
                if cleaned_preds is None:
                    _, cleaned_preds = _presence_rows(
                        image_id=image_id,
                        gt_mask=mask,
                        probabilities=probs,
                        threshold=threshold,
                        min_component_pixels=min_component_pixels,
                    )
                save_prediction(
                    image.cpu(),
                    mask.cpu(),
                    cleaned_preds.cpu(),
                    save_dir=str(save_dir),
                    image_id=image_id,
                )

    avg_class_dice = [
        None if count == 0 else float(dice_sum / count)
        for dice_sum, count in zip(class_dice_sum, class_valid_count)
    ]
    valid_scores = [score for score in avg_class_dice if score is not None]
    mean_dice = float(np.mean(valid_scores)) if valid_scores else float("nan")

    if not return_details:
        return mean_dice, avg_class_dice, class_valid_count.tolist()

    details = {
        "mean_dice": mean_dice,
        "class_dice": avg_class_dice,
        "class_counts": class_valid_count.tolist(),
        "presence_per_image": pd.DataFrame(presence_records),
        "presence_summary": pd.DataFrame(),
    }

    if collect_presence and presence_records:
        per_image = details["presence_per_image"]
        summary = (
            per_image.groupby("structure", sort=False)[
                ["gt_present", "pred_present", "tp", "fp", "fn", "tn"]
            ]
            .sum()
            .reset_index()
        )
        summary["total_errors"] = summary["fp"] + summary["fn"]
        details["presence_summary"] = summary

    return details
