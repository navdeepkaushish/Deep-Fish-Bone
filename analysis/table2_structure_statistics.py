"""
Generate CMPB Table 2: structure-level descriptive statistics.

For each of the 25 annotated skeletal structures, this script computes:

1. Occurrence frequency:
   - Number of images in which the structure is present.
   - Percentage of dataset images in which the structure is present.

2. Mean relative area:
   - Mask area divided by the corresponding original image area.
   - Expressed as a percentage.
   - Calculated only over images in which the structure is present.

3. Coefficient of variation:
   - Standard deviation of relative area divided by its mean.
   - Calculated only over images in which the structure is present.

Expected dataset structure
--------------------------
deep-fish-bone/
└── ventral/
    ├── <image_id>/
    │   ├── image.png
    │   ├── <structure_name>.png
    │   ├── ...
    │   ├── VC.png
    │   └── full_mask.png
    └── ...

Outputs
-------
paper_results/tables/table2_structure_statistics.csv
paper_results/tables/table2_structure_statistics.tex
paper_results/metadata/structure_statistics.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "ventral"

TABLES_DIR = PROJECT_ROOT / "tables"
TABLE_OUTPUT_DIR = TABLES_DIR
METADATA_OUTPUT_DIR = TABLES_DIR

TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------

IMAGE_FILENAME = "image.png"
FULL_MASK_FILENAME = "full_mask.png"

EXPECTED_NUMBER_OF_STRUCTURES = 25

# Expected original image dimensions:
# width = 2576, height = 1932
EXPECTED_IMAGE_WIDTH = 2576
EXPECTED_IMAGE_HEIGHT = 1932

VALID_MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}

# Files that must not be interpreted as individual structure masks.
EXCLUDED_FILENAMES = {
    IMAGE_FILENAME.lower(),
    FULL_MASK_FILENAME.lower(),
}


# ---------------------------------------------------------------------
# Optional manuscript structure order
# ---------------------------------------------------------------------
#
# When set to None, structures are sorted alphabetically, with VC placed
# at the end.
#
# To use a fixed anatomical order, replace None with the exact list of
# 25 mask filenames without the ".png" extension.
#
# Example:
#
# STRUCTURE_ORDER = [
#     "Br1a", "Br1b",
#     ...
#     "VC",
# ]
#
STRUCTURE_ORDER: list[str] | None = None


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def read_grayscale_image(path: Path) -> np.ndarray:
    """Read a mask or image as a two-dimensional grayscale array."""

    array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if array is None:
        raise RuntimeError(f"Could not read image file: {path}")

    if array.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional grayscale array for {path}, "
            f"but received shape {array.shape}."
        )

    return array


def discover_image_directories(dataset_root: Path) -> list[Path]:
    """Return folders that contain an image.png file."""

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_root}"
        )

    image_directories = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / IMAGE_FILENAME).exists()
    )

    if not image_directories:
        raise RuntimeError(
            f"No image-ID folders containing '{IMAGE_FILENAME}' were "
            f"found in {dataset_root}."
        )

    return image_directories


def discover_structure_masks(image_directory: Path) -> dict[str, Path]:
    """
    Discover individual structure masks in one image-ID directory.

    image.png and full_mask.png are excluded.
    """

    structure_masks: dict[str, Path] = {}

    for file_path in sorted(image_directory.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in VALID_MASK_EXTENSIONS:
            continue

        if file_path.name.lower() in EXCLUDED_FILENAMES:
            continue

        structure_name = file_path.stem

        if structure_name in structure_masks:
            raise RuntimeError(
                f"Duplicate mask detected for structure '{structure_name}' "
                f"in {image_directory}."
            )

        structure_masks[structure_name] = file_path

    return structure_masks


def sort_structure_names(structure_names: set[str]) -> list[str]:
    """Return structures in manuscript or alphabetical order."""

    if STRUCTURE_ORDER is not None:
        expected = set(STRUCTURE_ORDER)

        missing_from_order = structure_names - expected
        absent_from_dataset = expected - structure_names

        if missing_from_order or absent_from_dataset:
            raise ValueError(
                "STRUCTURE_ORDER does not match the discovered masks.\n"
                f"Discovered but missing from STRUCTURE_ORDER: "
                f"{sorted(missing_from_order)}\n"
                f"Listed but not found in dataset: "
                f"{sorted(absent_from_dataset)}"
            )

        return STRUCTURE_ORDER.copy()

    # Alphabetical ordering, but keep VC at the end.
    names = sorted(name for name in structure_names if name.upper() != "VC")

    vc_names = sorted(
        name for name in structure_names if name.upper() == "VC"
    )

    return names + vc_names


def coefficient_of_variation(values: list[float]) -> float:
    """
    Calculate CV = sample standard deviation / mean.

    NaN is returned when CV cannot be estimated reliably, for example
    when a structure occurs in only one image or has a zero mean.
    """

    if len(values) < 2:
        return float("nan")

    values_array = np.asarray(values, dtype=np.float64)
    mean_value = float(np.mean(values_array))

    if np.isclose(mean_value, 0.0):
        return float("nan")

    sample_std = float(np.std(values_array, ddof=1))

    return sample_std / mean_value


def format_latex_table(table: pd.DataFrame) -> str:
    """Create a compact LaTeX table suitable for Overleaf."""

    latex_table = table.to_latex(
        index=False,
        escape=True,
        na_rep="--",
        column_format="lrrrr",
        caption=(
            "Descriptive statistics of the twenty-five annotated skeletal "
            "structures, including occurrence frequency, mean relative area "
            "expressed as a percentage of the original image area, and "
            "coefficient of variation (CV). Mean relative area and CV were "
            "calculated using only images in which the corresponding "
            "structure was present."
        ),
        label="tab:structure_statistics",
        position="htbp",
        float_format="%.3f",
    )

    return latex_table


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------


def main() -> None:
    image_directories = discover_image_directories(DATASET_ROOT)
    number_of_images = len(image_directories)

    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Image-ID folders found: {number_of_images}")

    # Relative areas are stored only for present masks.
    relative_areas_by_structure: dict[str, list[float]] = defaultdict(list)

    # Occurrence count is the number of non-empty masks.
    occurrence_by_structure: dict[str, int] = defaultdict(int)

    all_structure_names: set[str] = set()
    reference_structure_names: set[str] | None = None

    unexpected_image_dimensions: list[tuple[str, int, int]] = []

    for image_index, image_directory in enumerate(image_directories, start=1):
        image_id = image_directory.name
        image_path = image_directory / IMAGE_FILENAME

        image = read_grayscale_image(image_path)
        image_height, image_width = image.shape

        if (
            image_width != EXPECTED_IMAGE_WIDTH
            or image_height != EXPECTED_IMAGE_HEIGHT
        ):
            unexpected_image_dimensions.append(
                (image_id, image_width, image_height)
            )

        image_area_pixels = image_height * image_width

        if image_area_pixels <= 0:
            raise ValueError(
                f"Invalid image dimensions for {image_path}: {image.shape}"
            )

        structure_masks = discover_structure_masks(image_directory)
        current_structure_names = set(structure_masks)

        if len(current_structure_names) != EXPECTED_NUMBER_OF_STRUCTURES:
            raise RuntimeError(
                f"{image_id} contains {len(current_structure_names)} "
                f"individual structure masks; expected "
                f"{EXPECTED_NUMBER_OF_STRUCTURES}.\n"
                f"Discovered masks: {sorted(current_structure_names)}"
            )

        if reference_structure_names is None:
            reference_structure_names = current_structure_names
        elif current_structure_names != reference_structure_names:
            missing = reference_structure_names - current_structure_names
            additional = current_structure_names - reference_structure_names

            raise RuntimeError(
                f"Inconsistent mask set in folder '{image_id}'.\n"
                f"Missing structures: {sorted(missing)}\n"
                f"Unexpected structures: {sorted(additional)}"
            )

        all_structure_names.update(current_structure_names)

        for structure_name, mask_path in structure_masks.items():
            mask = read_grayscale_image(mask_path)

            if mask.shape != image.shape:
                raise ValueError(
                    f"Dimension mismatch in folder '{image_id}':\n"
                    f"image.png shape: {image.shape}\n"
                    f"{mask_path.name} shape: {mask.shape}"
                )

            # Any positive pixel is treated as foreground.
            foreground_pixels = int(np.count_nonzero(mask > 0))

            # An all-zero mask indicates that the structure is absent.
            if foreground_pixels == 0:
                continue

            occurrence_by_structure[structure_name] += 1

            relative_area_percentage = (
                foreground_pixels / image_area_pixels
            ) * 100.0

            relative_areas_by_structure[structure_name].append(
                relative_area_percentage
            )

        print(
            f"\rProcessed {image_index}/{number_of_images} folders",
            end="",
            flush=True,
        )

    print()

    if len(all_structure_names) != EXPECTED_NUMBER_OF_STRUCTURES:
        raise RuntimeError(
            f"Found {len(all_structure_names)} unique structures across the "
            f"dataset; expected {EXPECTED_NUMBER_OF_STRUCTURES}.\n"
            f"Structures found: {sorted(all_structure_names)}"
        )

    structure_names = sort_structure_names(all_structure_names)

    # -----------------------------------------------------------------
    # Create Table 2
    # -----------------------------------------------------------------

    rows: list[dict[str, object]] = []
    detailed_json: dict[str, dict[str, object]] = {}

    for structure_name in structure_names:
        areas = relative_areas_by_structure.get(structure_name, [])
        occurrence_count = occurrence_by_structure.get(structure_name, 0)

        occurrence_percentage = (
            occurrence_count / number_of_images
        ) * 100.0

        mean_relative_area = (
            float(np.mean(areas)) if areas else float("nan")
        )

        cv = coefficient_of_variation(areas)

        rows.append(
            {
                "Structure": structure_name,
                "Occurrence (n)": occurrence_count,
                "Occurrence (%)": occurrence_percentage,
                "Mean relative area (%)": mean_relative_area,
                "CV": cv,
            }
        )

        detailed_json[structure_name] = {
            "occurrence_count": int(occurrence_count),
            "occurrence_percentage": float(occurrence_percentage),
            "mean_relative_area_percentage": (
                None
                if np.isnan(mean_relative_area)
                else float(mean_relative_area)
            ),
            "coefficient_of_variation": (
                None if np.isnan(cv) else float(cv)
            ),
            "number_of_present_masks_used_for_area": len(areas),
        }

    raw_table = pd.DataFrame(rows)

    # Publication-ready rounded version.
    publication_table = raw_table.copy()

    publication_table["Occurrence (%)"] = publication_table[
        "Occurrence (%)"
    ].round(1)

    publication_table["Mean relative area (%)"] = publication_table[
        "Mean relative area (%)"
    ].round(3)

    publication_table["CV"] = publication_table["CV"].round(3)

    # -----------------------------------------------------------------
    # Consistency checks
    # -----------------------------------------------------------------

    for _, row in raw_table.iterrows():
        structure = str(row["Structure"])
        occurrence = int(row["Occurrence (n)"])
        area_count = len(relative_areas_by_structure.get(structure, []))

        if occurrence != area_count:
            raise AssertionError(
                f"Occurrence/area inconsistency for {structure}: "
                f"occurrence={occurrence}, area values={area_count}."
            )

        if occurrence > number_of_images:
            raise AssertionError(
                f"Occurrence count for {structure} exceeds the number "
                f"of dataset images."
            )

    # -----------------------------------------------------------------
    # Save files
    # -----------------------------------------------------------------

    csv_path = TABLE_OUTPUT_DIR / "table2_structure_statistics.csv"
    raw_csv_path = (
        TABLE_OUTPUT_DIR / "table2_structure_statistics_unrounded.csv"
    )
    latex_path = TABLE_OUTPUT_DIR / "table2_structure_statistics.tex"
    json_path = (
        METADATA_OUTPUT_DIR / "structure_statistics.json"
    )

    publication_table.to_csv(csv_path, index=False)
    raw_table.to_csv(raw_csv_path, index=False)

    latex_text = format_latex_table(publication_table)
    latex_path.write_text(latex_text, encoding="utf-8")

    json_output = {
        "dataset_root": str(DATASET_ROOT),
        "number_of_images": number_of_images,
        "number_of_structures": len(structure_names),
        "relative_area_definition": (
            "100 * foreground mask pixels / pixels in the corresponding "
            "original image"
        ),
        "area_statistics_include_absent_masks": False,
        "cv_definition": (
            "sample standard deviation of present-mask relative areas "
            "divided by their mean"
        ),
        "structures": detailed_json,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(json_output, file, indent=2)

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    print("\nTable 2 generated successfully.\n")
    print(publication_table.to_string(index=False))

    print("\nSaved files:")
    print(f"  {csv_path}")
    print(f"  {raw_csv_path}")
    print(f"  {latex_path}")
    print(f"  {json_path}")

    if unexpected_image_dimensions:
        print(
            "\nWARNING: The following image.png files did not have the "
            "expected original dimensions of "
            f"{EXPECTED_IMAGE_WIDTH} × {EXPECTED_IMAGE_HEIGHT}:"
        )

        for image_id, width, height in unexpected_image_dimensions:
            print(f"  {image_id}: {width} × {height}")

        print(
            "\nRelative area remains calculated using each image's actual "
            "pixel dimensions. Verify these folders before finalizing the "
            "manuscript."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)