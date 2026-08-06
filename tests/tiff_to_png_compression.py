from pathlib import Path
import csv
import shutil
import sys

import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# Example structure:
#
# dataset_root/
# ├── ventral_all/
# │   ├── 163428021.tif
# │   ├── 163428022.tif
# │   └── ...
# ├── ventral/
# │   ├── 163428021/
# │   │   ├── image.png
# │   │   ├── Br2a.png
# │   │   ├── VC.png
# │   │   └── ...
# │   └── ...
# └── png_compressed/
#     ├── 163428021/
#     │   ├── image.png
#     │   ├── Br2a.png
#     │   ├── VC.png
#     │   └── ...

ROOT_DIR = Path("/Users/navdeepkaushish/Documents/missing_bone_structures")

TIFF_DIR = ROOT_DIR / "ventral_all"
SOURCE_DATASET_DIR = ROOT_DIR / "ventral"
OUTPUT_DIR = ROOT_DIR / "png_compressed"

# PNG compression level:
# 0 = no compression
# 9 = maximum lossless compression
PNG_COMPRESS_LEVEL = 9

# Additional PNG optimization.
# This can reduce file size further but makes conversion slower.
PNG_OPTIMIZE = True

# Verify that TIFF and compressed PNG decode to exactly equal arrays.
VERIFY_PIXEL_EQUALITY = True

# Copy all files from the original image-ID folder except image.png.
# Normally these files will be annotation masks.
COPY_ALL_NON_IMAGE_FILES = True

# If False, an existing output image-ID folder will be skipped.
# If True, files inside it may be replaced.
OVERWRITE_EXISTING = False


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def load_single_frame_image(path: Path) -> tuple[np.ndarray, str]:
    """
    Load one TIFF or PNG image without intentionally changing its
    channels, intensity range, bit depth, or pixel values.

    Returns
    -------
    array:
        Decoded image as a NumPy array.
    mode:
        Original Pillow image mode.
    """
    with Image.open(path) as image:
        number_of_frames = getattr(image, "n_frames", 1)

        if number_of_frames != 1:
            raise ValueError(
                f"{path.name} contains {number_of_frames} frames. "
                "This script expects a single-frame TIFF."
            )

        image.load()
        mode = image.mode
        array = np.asarray(image).copy()

    return array, mode


def save_compressed_png(
    source_tiff: Path,
    destination_png: Path,
) -> dict:
    """
    Convert a TIFF into a losslessly compressed PNG and optionally
    verify exact decoded-pixel equality.
    """
    source_array, source_mode = load_single_frame_image(source_tiff)

    destination_png.parent.mkdir(parents=True, exist_ok=True)

    # Reconstruct the Pillow image from the exact decoded array.
    output_image = Image.fromarray(source_array)

    output_image.save(
        destination_png,
        format="PNG",
        compress_level=PNG_COMPRESS_LEVEL,
        optimize=PNG_OPTIMIZE,
    )

    result = {
        "source_mode": source_mode,
        "source_shape": str(source_array.shape),
        "source_dtype": str(source_array.dtype),
        "source_min": float(source_array.min()),
        "source_max": float(source_array.max()),
        "pixels_identical": "",
        "max_pixel_difference": "",
    }

    if VERIFY_PIXEL_EQUALITY:
        output_array, output_mode = load_single_frame_image(destination_png)

        same_shape = source_array.shape == output_array.shape
        same_dtype = source_array.dtype == output_array.dtype
        pixels_identical = (
            same_shape
            and same_dtype
            and np.array_equal(source_array, output_array)
        )

        result["output_mode"] = output_mode
        result["pixels_identical"] = pixels_identical

        if same_shape:
            difference = np.abs(
                source_array.astype(np.float64)
                - output_array.astype(np.float64)
            )
            result["max_pixel_difference"] = float(difference.max())
        else:
            result["max_pixel_difference"] = "shape mismatch"

        if not pixels_identical:
            destination_png.unlink(missing_ok=True)

            raise ValueError(
                "Pixel verification failed: "
                f"shape {source_array.shape} vs {output_array.shape}, "
                f"dtype {source_array.dtype} vs {output_array.dtype}, "
                f"mode {source_mode} vs {output_mode}."
            )

    return result


# ============================================================
# MASK-COPYING FUNCTIONS
# ============================================================

def copy_annotation_files(
    source_folder: Path,
    destination_folder: Path,
) -> int:
    """
    Copy annotation files from ventral/<image_id>/ into the newly
    created output folder.

    The existing source image.png is deliberately excluded.
    """
    copied_count = 0

    for source_item in sorted(source_folder.iterdir()):
        # Do not copy the old, approximately 14 MB input image.
        if source_item.is_file() and source_item.name.lower() == "image.png":
            continue

        destination_item = destination_folder / source_item.name

        if source_item.is_file():
            # By default, copy all non-image files, including every mask.
            if COPY_ALL_NON_IMAGE_FILES:
                shutil.copy2(source_item, destination_item)
                copied_count += 1

            # Alternatively, restrict copying to PNG masks.
            elif source_item.suffix.lower() == ".png":
                shutil.copy2(source_item, destination_item)
                copied_count += 1

        elif source_item.is_dir():
            # This is included for robustness in case an image-ID folder
            # contains nested annotation directories.
            shutil.copytree(
                source_item,
                destination_item,
                dirs_exist_ok=OVERWRITE_EXISTING,
            )

    return copied_count


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def find_tiff_files(folder: Path) -> list[Path]:
    """Find both .tif and .tiff files, case-insensitively."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
    )


def size_in_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def validate_directories() -> None:
    if not TIFF_DIR.exists():
        raise FileNotFoundError(
            f"TIFF directory does not exist:\n{TIFF_DIR.resolve()}"
        )

    if not SOURCE_DATASET_DIR.exists():
        raise FileNotFoundError(
            "Original ventral dataset directory does not exist:\n"
            f"{SOURCE_DATASET_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN CONVERSION
# ============================================================

def main() -> None:
    validate_directories()

    tiff_files = find_tiff_files(TIFF_DIR)

    if not tiff_files:
        raise FileNotFoundError(
            f"No .tif or .tiff files were found in:\n{TIFF_DIR.resolve()}"
        )

    print("=" * 80)
    print("TIFF TO COMPRESSED PNG DATASET CONVERSION")
    print("=" * 80)
    print(f"TIFF source       : {TIFF_DIR.resolve()}")
    print(f"Mask source       : {SOURCE_DATASET_DIR.resolve()}")
    print(f"Output dataset    : {OUTPUT_DIR.resolve()}")
    print(f"TIFF files found  : {len(tiff_files)}")
    print(f"PNG compression   : {PNG_COMPRESS_LEVEL}")
    print(f"PNG optimization  : {PNG_OPTIMIZE}")
    print(f"Pixel verification: {VERIFY_PIXEL_EQUALITY}")
    print("=" * 80)

    report_rows = []

    successful = 0
    skipped = 0
    failed = 0
    missing_mask_folders = 0

    total_tiff_size_mb = 0.0
    total_png_size_mb = 0.0

    for index, tiff_path in enumerate(tiff_files, start=1):
        image_id = tiff_path.stem

        source_mask_folder = SOURCE_DATASET_DIR / image_id
        output_image_folder = OUTPUT_DIR / image_id
        output_png_path = output_image_folder / "image.png"

        print(
            f"\n[{index:03d}/{len(tiff_files):03d}] "
            f"Processing image ID: {image_id}"
        )

        report = {
            "image_id": image_id,
            "tiff_file": tiff_path.name,
            "source_mask_folder": str(source_mask_folder),
            "output_folder": str(output_image_folder),
            "status": "",
            "message": "",
            "tiff_size_mb": "",
            "compressed_png_size_mb": "",
            "size_reduction_mb": "",
            "size_reduction_percent": "",
            "number_of_masks_copied": 0,
            "source_mode": "",
            "output_mode": "",
            "source_shape": "",
            "source_dtype": "",
            "source_min": "",
            "source_max": "",
            "pixels_identical": "",
            "max_pixel_difference": "",
        }

        try:
            tiff_size = size_in_mb(tiff_path)
            report["tiff_size_mb"] = round(tiff_size, 4)

            if output_image_folder.exists() and not OVERWRITE_EXISTING:
                report["status"] = "skipped"
                report["message"] = (
                    "Output folder already exists and "
                    "OVERWRITE_EXISTING=False."
                )

                print("  SKIPPED: output folder already exists.")
                skipped += 1
                report_rows.append(report)
                continue

            output_image_folder.mkdir(parents=True, exist_ok=True)

            conversion_result = save_compressed_png(
                source_tiff=tiff_path,
                destination_png=output_png_path,
            )

            png_size = size_in_mb(output_png_path)
            reduction_mb = tiff_size - png_size
            reduction_percent = (
                100.0 * reduction_mb / tiff_size
                if tiff_size > 0
                else 0.0
            )

            report["compressed_png_size_mb"] = round(png_size, 4)
            report["size_reduction_mb"] = round(reduction_mb, 4)
            report["size_reduction_percent"] = round(
                reduction_percent, 2
            )

            report.update(conversion_result)

            if not source_mask_folder.exists():
                missing_mask_folders += 1
                report["status"] = "partial_success"
                report["message"] = (
                    "Compressed image created, but corresponding "
                    "ventral image-ID folder was not found."
                )

                print("  PNG created successfully.")
                print(
                    "  WARNING: source mask folder not found:"
                    f" {source_mask_folder}"
                )
            else:
                number_of_masks = copy_annotation_files(
                    source_folder=source_mask_folder,
                    destination_folder=output_image_folder,
                )

                report["number_of_masks_copied"] = number_of_masks
                report["status"] = "success"
                report["message"] = (
                    "Compressed PNG created and annotation files copied."
                )

                print(f"  Masks/files copied : {number_of_masks}")

            print(f"  TIFF size          : {tiff_size:.3f} MB")
            print(f"  Compressed PNG size: {png_size:.3f} MB")
            print(f"  Size reduction     : {reduction_percent:.2f}%")

            if VERIFY_PIXEL_EQUALITY:
                print(
                    "  Pixel-identical     : "
                    f"{conversion_result['pixels_identical']}"
                )

            total_tiff_size_mb += tiff_size
            total_png_size_mb += png_size
            successful += 1

        except Exception as error:
            failed += 1
            report["status"] = "failed"
            report["message"] = str(error)

            print(f"  FAILED: {error}")

            # Remove the incomplete output folder if conversion failed.
            if output_image_folder.exists():
                shutil.rmtree(output_image_folder)

        report_rows.append(report)

    # --------------------------------------------------------
    # Save CSV report
    # --------------------------------------------------------

    report_path = OUTPUT_DIR / "compression_and_copy_report.csv"

    fieldnames = list(report_rows[0].keys())

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"Total TIFF files           : {len(tiff_files)}")
    print(f"Successfully converted     : {successful}")
    print(f"Skipped                    : {skipped}")
    print(f"Failed                     : {failed}")
    print(f"Missing mask folders       : {missing_mask_folders}")
    print(f"Original TIFF total        : {total_tiff_size_mb:.2f} MB")
    print(f"Compressed PNG total       : {total_png_size_mb:.2f} MB")

    if total_tiff_size_mb > 0:
        total_saved_mb = total_tiff_size_mb - total_png_size_mb
        total_reduction = (
            100.0 * total_saved_mb / total_tiff_size_mb
        )

        print(f"Total space saved          : {total_saved_mb:.2f} MB")
        print(f"Overall reduction          : {total_reduction:.2f}%")

    print(f"Output directory           : {OUTPUT_DIR.resolve()}")
    print(f"CSV report                 : {report_path.resolve()}")
    print("=" * 80)

    if failed > 0:
        print(
            "\nSome files failed. Check the CSV report before using "
            "the new dataset."
        )
        sys.exit(1)

    print(
        "\nConversion completed. The new dataset should be used only "
        "after confirming that every expected image ID and mask is present."
    )


if __name__ == "__main__":
    main()