"""
Prepare train/val CSVs for the Nutrition5k portion estimator.

Reads the raw `metadata/dish_metadata_cafe{1,2}.csv` files (variable-length rows:
dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, num_ingrs, ...),
extracts dish_id + total_mass, verifies the RGB image exists, and writes:

    data/nutrition5k/train.csv
    data/nutrition5k/val.csv

with columns `image_name,weight_g` where `image_name` is relative to
`data/nutrition5k/imagery/realsense_overhead/`.

Usage:
    python scripts/prepare_nutrition5k.py
"""

import csv
import random
from pathlib import Path

DATA_DIR = Path("./data/nutrition5k")
META_DIR = DATA_DIR / "metadata"
IMG_ROOT = DATA_DIR / "imagery" / "realsense_overhead"
VAL_FRACTION = 0.1
RANDOM_SEED = 42


def parse_metadata(csv_path: Path):
    """Yield (dish_id, total_mass_g) from a Nutrition5k dish_metadata CSV."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            dish_id = row[0].strip()
            try:
                total_mass = float(row[2])
            except ValueError:
                continue
            if total_mass <= 0:
                continue
            yield dish_id, total_mass


def main():
    if not META_DIR.exists():
        raise SystemExit(f"Metadata dir not found: {META_DIR}. Run download_nutrition5k.py first.")
    if not IMG_ROOT.exists():
        raise SystemExit(f"Imagery dir not found: {IMG_ROOT}. Run download_nutrition5k.py first.")

    rows = []
    skipped_no_img = 0
    for cafe_csv in sorted(META_DIR.glob("dish_metadata_cafe*.csv")):
        for dish_id, weight in parse_metadata(cafe_csv):
            rel = f"{dish_id}/rgb.png"
            if not (IMG_ROOT / rel).exists():
                skipped_no_img += 1
                continue
            rows.append((rel, weight))

    if not rows:
        raise SystemExit("No usable (image, weight) pairs found.")

    random.seed(RANDOM_SEED)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_FRACTION))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    for name, subset in [("train.csv", train_rows), ("val.csv", val_rows)]:
        out = DATA_DIR / name
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_name", "weight_g"])
            w.writerows(subset)
        print(f"Wrote {out} ({len(subset)} rows)")

    print(f"Skipped {skipped_no_img} dishes with no rgb.png on disk.")
    print(f"Use: --img_dir {IMG_ROOT}")


if __name__ == "__main__":
    main()
