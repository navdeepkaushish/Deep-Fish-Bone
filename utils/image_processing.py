from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import re

# ============================================================
# Paths
# ============================================================

source_root = Path(r"/Users/navdeepkaushish/Documents/missing_bone_structures/dataset_new/ventral")
output_root = Path(r"/Users/navdeepkaushish/Documents/missing_bone_structures/dataset_new/processed_images")
output_root.mkdir(parents=True, exist_ok=True)

# ============================================================
# FULL MASK DETECTOR (FIXED LOGIC)
# ============================================================

def is_full_mask(filename: str) -> bool:
    """
    ONLY checks suffix:
    - .tif.png
    - .tif png (with any whitespace)
    """

    name = filename.lower()

    # normalize whitespace (tabs, multiple spaces)
    name = re.sub(r"\s+", " ", name)

    # remove spaces before checking suffix
    name_no_space = name.replace(" ", "")

    return name_no_space.endswith(".tif.png")

# ============================================================
# VC detector
# ============================================================

def is_vc_mask(stem: str) -> bool:
    stem = stem.upper().strip()
    return bool(re.match(r"VC\d+$", stem))

# ============================================================
# MAIN LOOP
# ============================================================

for folder in sorted(source_root.iterdir()):

    if not folder.is_dir():
        continue

    img_id = folder.name
    print(f"\nProcessing {img_id}")

    out_dir = output_root / img_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. TIFF image -> image.png
    # --------------------------------------------------------

    tif_files = list(folder.glob("*.tif"))

    if not tif_files:
        print("  No TIFF image found")
        continue

    image_file = sorted(tif_files)[0]
    shutil.copy2(image_file, out_dir / "image.png")

    # --------------------------------------------------------
    # 2. FULL MASK (FIXED - ONLY CHECKS SUFFIX)
    # --------------------------------------------------------

    full_mask = None

    for f in folder.glob("*.png"):

        normalized = re.sub(r"\s+", "", f.name.lower())

        if normalized.endswith(".tif.png"):
            full_mask = f
            break

    if full_mask:
        shutil.copy2(full_mask, out_dir / "full_mask.png")
    else:
        print("  WARNING: full mask not found")

    # --------------------------------------------------------
    # 3. Separate VC masks
    # --------------------------------------------------------

    vc_masks = []
    other_masks = []

    for f in folder.glob("*.png"):

        if full_mask and f == full_mask:
            continue

        if is_vc_mask(f.stem):
            vc_masks.append(f)
        else:
            other_masks.append(f)

    # --------------------------------------------------------
    # 4. Merge VC masks
    # --------------------------------------------------------

    if vc_masks:

        merged = None

        for f in sorted(vc_masks):

            arr = np.array(Image.open(f).convert("L")) > 0

            merged = arr if merged is None else (merged | arr)

        Image.fromarray((merged.astype(np.uint8) * 255)).save(
            out_dir / "VC.png"
        )

    # --------------------------------------------------------
    # 5. Copy remaining masks
    # --------------------------------------------------------

    for f in other_masks:
        shutil.copy2(f, out_dir / f.name)

    print(f"  image: {image_file.name}")
    print(f"  full mask: {full_mask.name if full_mask else 'NOT FOUND'}")
    print(f"  VC merged: {len(vc_masks)}")
    print(f"  other masks: {len(other_masks)}")

print("\nDONE")