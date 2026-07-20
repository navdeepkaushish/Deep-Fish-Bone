#!/usr/bin/env python3
"""Generate model-assisted masks for suspected missing skeletal structures.

Expected input layout
---------------------
Missing_Structures/
    Oc1/
        image_001.png
        image_002.png
    CH1/
        image_010.png

The name of each immediate subfolder must match one entry in STRUCTURES.
Only that structure's output channel is saved for each image.

Run from the deep-fish-bone repository, for example:

python generate_missing_structure_masks.py

or override paths/settings:

python generate_missing_structure_masks.py \
    --input-root /path/to/Missing_Structures \
    --checkpoint /path/to/best_model.pt \
    --output-root /path/to/Predicted_Missing_Structures
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from configs.structure_names import STRUCTURES
from inference.sliding_window import sliding_window_predict
from models.unetpp import build_model

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise SystemExit(
        "segmentation_models_pytorch is not installed. Activate the same "
        "environment used for training first."
    ) from exc



IMAGE_EXTENSIONS = {".png"}

# Defaults based on the paths supplied by the user.
DEFAULT_PROJECT_ROOT = Path(
    "/Users/navdeepkaushish/Documents/missing_bone_structures/"
    "deep-fish-bone"
)
DEFAULT_CHECKPOINT = (
    DEFAULT_PROJECT_ROOT
    / "outputs_cv_focal/fold_2/checkpoints/best_model.pt"
)
DEFAULT_INPUT_ROOT = DEFAULT_PROJECT_ROOT / "Missing_Structures"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "Predicted_Missing_Structures"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained 25-class model on structure-named folders and "
            "save only the corresponding predicted structure mask."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-pixels", type=int, default=25)
    parser.add_argument(
        "--inference-width",
        type=int,
        default=1288,
        help="Width used during full-image inference; set 0 to keep original size.",
    )
    parser.add_argument(
        "--inference-height",
        type=int,
        default=966,
        help="Height used during full-image inference; set 0 to keep original size.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--save-probability",
        action="store_true",
        help="Also save a 16-bit probability map for later threshold adjustment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask/overlay files.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")




def _looks_like_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and value and all(
        isinstance(k, str) for k in value.keys()
    ) and any(torch.is_tensor(v) for v in value.values())


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Support common checkpoint formats used by PyTorch training scripts."""
    if _looks_like_state_dict(checkpoint):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict):
        candidate_keys = (
            "model_state_dict", "state_dict", "model", "network", "net"
        )
        state_dict = None
        for key in candidate_keys:
            candidate = checkpoint.get(key)
            if _looks_like_state_dict(candidate):
                state_dict = candidate
                break
        if state_dict is None:
            raise ValueError(
                "Could not find model weights in checkpoint. Available keys: "
                f"{list(checkpoint.keys())}"
            )
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    # Handle checkpoints saved from DataParallel/DDP or wrapped model objects.
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights do not exactly match the expected UNet++ "
            "ResNet-34, 3-input, 25-output architecture."
        ) from exc

    model.to(device)
    model.eval()
    return model


def read_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(
    image_rgb: np.ndarray,
    inference_width: int,
    inference_height: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    original_height, original_width = image_rgb.shape[:2]

    if inference_width > 0 and inference_height > 0:
        resized = cv2.resize(
            image_rgb,
            (inference_width, inference_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image_rgb

    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor.contiguous(), (original_height, original_width)




def remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if min_pixels <= 1 or not mask.any():
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= min_pixels:
            cleaned[labels == label_index] = 1
    return cleaned


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a visible red overlay without changing unmasked pixels."""
    overlay = image_rgb.copy()
    red = np.zeros_like(image_rgb)
    red[..., 0] = 255
    alpha = 0.45
    selected = mask.astype(bool)
    overlay[selected] = (
        (1.0 - alpha) * image_rgb[selected] + alpha * red[selected]
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 0, 255), 2)
    return overlay_bgr


def canonical_structure_name(folder_name: str) -> str | None:
    mapping = {name.casefold(): name for name in STRUCTURES}
    return mapping.get(folder_name.casefold())


def collect_jobs(input_root: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    jobs: list[tuple[str, Path]] = []
    warnings: list[str] = []

    for folder in sorted(path for path in input_root.iterdir() if path.is_dir()):
        structure = canonical_structure_name(folder.name)
        if structure is None:
            warnings.append(
                f"Skipping unknown structure folder '{folder.name}'. "
                f"Valid names: {', '.join(STRUCTURES)}"
            )
            continue

        images = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            warnings.append(f"No supported images found in: {folder}")
            continue

        jobs.extend((structure, image_path) for image_path in images)

    return jobs, warnings


def main() -> int:
    args = parse_args()

    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    device = choose_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print("Input preprocessing: OpenCV PNG -> RGB -> resize -> float32 / 255")
    print("Sliding window: inference.sliding_window.sliding_window_predict")

    jobs, warnings = collect_jobs(args.input_root)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if not jobs:
        raise RuntimeError("No valid structure/image jobs were found.")

    model = load_model(args.checkpoint, device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    log_rows: list[dict[str, object]] = []

    for structure, image_path in tqdm(jobs, desc="Generating masks", unit="image"):
        output_dir = args.output_root / structure
        output_dir.mkdir(parents=True, exist_ok=True)

        relative_to_structure = image_path.relative_to(args.input_root / image_path.parts[len(args.input_root.parts)])
        # Keep image stems unique when nested source folders are used.
        safe_relative_stem = "__".join(relative_to_structure.with_suffix("").parts)

        mask_path = output_dir / f"{safe_relative_stem}_mask.png"
        overlay_path = output_dir / f"{safe_relative_stem}_overlay.png"
        probability_path = output_dir / f"{safe_relative_stem}_probability.tif"

        if mask_path.exists() and overlay_path.exists() and not args.overwrite:
            log_rows.append({
                "structure": structure,
                "class_index": STRUCTURES.index(structure),
                "source_image": str(image_path),
                "status": "skipped_existing",
                "threshold": args.threshold,
                "max_probability": "",
                "mean_probability_in_mask": "",
                "predicted_pixels": "",
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "probability_path": str(probability_path) if args.save_probability else "",
            })
            continue

        try:
            image_rgb = read_rgb_image(image_path)
            image_tensor, original_shape = preprocess_image(
                image_rgb,
                args.inference_width,
                args.inference_height,
            )

            all_probabilities = sliding_window_predict(
                model=model,
                image=image_tensor,
                device=device,
                patch_size=args.patch_size,
                stride=args.stride,
            )

            class_index = STRUCTURES.index(structure)
            probability = all_probabilities[class_index].numpy()

            original_height, original_width = original_shape
            if probability.shape != (original_height, original_width):
                probability_original = cv2.resize(
                    probability,
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                probability_original = probability

            binary_mask = (probability_original >= args.threshold).astype(np.uint8)
            binary_mask = remove_small_components(
                binary_mask, args.min_component_pixels
            )
            mask_uint8 = binary_mask * 255

            if not cv2.imwrite(str(mask_path), mask_uint8):
                raise IOError(f"Could not write mask: {mask_path}")

            overlay_bgr = make_overlay(image_rgb, binary_mask)
            if not cv2.imwrite(str(overlay_path), overlay_bgr):
                raise IOError(f"Could not write overlay: {overlay_path}")

            if args.save_probability:
                probability_uint16 = np.clip(
                    probability_original * 65535.0, 0, 65535
                ).astype(np.uint16)
                if not cv2.imwrite(str(probability_path), probability_uint16):
                    raise IOError(
                        f"Could not write probability map: {probability_path}"
                    )

            selected_probabilities = probability_original[binary_mask.astype(bool)]
            mean_in_mask = (
                float(selected_probabilities.mean())
                if selected_probabilities.size
                else 0.0
            )

            log_rows.append({
                "structure": structure,
                "class_index": class_index,
                "source_image": str(image_path),
                "status": "saved",
                "threshold": args.threshold,
                "max_probability": float(probability_original.max()),
                "mean_probability_in_mask": mean_in_mask,
                "predicted_pixels": int(binary_mask.sum()),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "probability_path": str(probability_path) if args.save_probability else "",
            })

        except Exception as exc:  # continue processing other images
            log_rows.append({
                "structure": structure,
                "class_index": STRUCTURES.index(structure),
                "source_image": str(image_path),
                "status": f"error: {exc}",
                "threshold": args.threshold,
                "max_probability": "",
                "mean_probability_in_mask": "",
                "predicted_pixels": "",
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "probability_path": str(probability_path) if args.save_probability else "",
            })
            print(f"ERROR processing {image_path}: {exc}", file=sys.stderr)

    log_path = args.output_root / "prediction_log.csv"
    fieldnames = [
        "structure", "class_index", "source_image", "status", "threshold",
        "max_probability", "mean_probability_in_mask", "predicted_pixels",
        "mask_path", "overlay_path", "probability_path",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    saved = sum(row["status"] == "saved" for row in log_rows)
    skipped = sum(row["status"] == "skipped_existing" for row in log_rows)
    errors = len(log_rows) - saved - skipped
    print("\nFinished")
    print(f"Saved: {saved}")
    print(f"Skipped existing: {skipped}")
    print(f"Errors: {errors}")
    print(f"Log: {log_path}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
