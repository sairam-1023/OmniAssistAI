"""
Splits data/images/<category>/*.png into data/images_split/{train,val}/<category>/
using an 80/20 split, so PyTorch's ImageFolder can consume it directly.
"""

import random
import shutil
from pathlib import Path

random.seed(42)

SRC_DIR = Path("data/images")
DST_DIR = Path("data/images_split")
VAL_FRACTION = 0.2


def main():
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)

    for category_dir in sorted(SRC_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        images = sorted(category_dir.glob("*.png"))
        random.shuffle(images)

        n_val = int(len(images) * VAL_FRACTION)
        val_images = images[:n_val]
        train_images = images[n_val:]

        for split_name, split_images in [("train", train_images), ("val", val_images)]:
            out_dir = DST_DIR / split_name / category
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy(img_path, out_dir / img_path.name)

        print(f"{category}: {len(train_images)} train, {len(val_images)} val")


if __name__ == "__main__":
    main()