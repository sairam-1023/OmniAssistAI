"""
Splits data/real_images/<category>/*.* into:
  - data/real_images_split/train/<category>/  (used for fine-tuning)
  - data/real_images_split/eval/<category>/   (held back, ONLY for honest evaluation)

Uses a minimum eval count per category since some categories have very
few real images to begin with.
"""

import random
import shutil
from pathlib import Path

random.seed(42)

SRC_DIR = Path("data/real_images")
DST_DIR = Path("data/real_images_split")
MIN_EVAL_PER_CLASS = 3
EVAL_FRACTION = 0.25


def main():
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)

    for category_dir in sorted(SRC_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        images = sorted([p for p in category_dir.iterdir() if p.is_file()])
        random.shuffle(images)

        n_eval = max(MIN_EVAL_PER_CLASS, int(len(images) * EVAL_FRACTION))
        n_eval = min(n_eval, len(images) - 2)  # always leave at least 2 for training

        eval_images = images[:n_eval]
        train_images = images[n_eval:]

        for split_name, split_images in [("train", train_images), ("eval", eval_images)]:
            out_dir = DST_DIR / split_name / category
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy(img_path, out_dir / img_path.name)

        print(f"{category}: {len(train_images)} train, {len(eval_images)} eval")


if __name__ == "__main__":
    main()