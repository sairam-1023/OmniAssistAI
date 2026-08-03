"""
Merges real training images into the existing synthetic train split,
so the model trains on a combination of both. Real eval images are
deliberately NOT touched here — they stay separate for honest testing.
"""

import shutil
from pathlib import Path

SYNTHETIC_TRAIN = Path("data/images_split/train")
REAL_TRAIN = Path("data/real_images_split/train")

for category_dir in sorted(REAL_TRAIN.iterdir()):
    if not category_dir.is_dir():
        continue
    category = category_dir.name
    dest_dir = SYNTHETIC_TRAIN / category
    dest_dir.mkdir(parents=True, exist_ok=True)

    for img_path in category_dir.iterdir():
        # Prefix filename to avoid collisions with synthetic images
        dest_path = dest_dir / f"real_{img_path.name}"
        shutil.copy(img_path, dest_path)

    print(f"Merged {len(list(category_dir.iterdir()))} real images into {dest_dir}")