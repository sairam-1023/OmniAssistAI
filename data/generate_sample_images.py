"""
Generates synthetic document images for training the Vision classifier.

Each category gets a visually distinct fake layout (not just noise) so
the model has real structure to learn from. Real scanned documents will
replace this later — this proves the training pipeline works honestly.
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw

random.seed(42)

CATEGORIES = ["invoice", "receipt", "id_card", "form", "prescription", "generic_photo"]
IMAGES_PER_CATEGORY = 120
OUT_DIR = Path("data/images")


def draw_invoice(draw, w, h):
    header_bottom = random.randint(60, 80)
    draw.rectangle([20, 20, w - 20, header_bottom], outline="black", width=2)
    y = header_bottom + random.randint(20, 35)
    for _ in range(random.randint(6, 10)):
        line_end = w - 30 - random.randint(0, 40)
        draw.line([30, y, line_end, y], fill="gray", width=1)
        y += random.randint(20, 28)


def draw_receipt(draw, w, h):
    y = random.randint(15, 25)
    for _ in range(random.randint(15, 22)):
        line_w = random.randint(60, w - 40)
        x_start = random.randint(15, 25)
        draw.line([x_start, y, x_start + line_w, y], fill="black", width=2)
        y += random.randint(12, 18)


def draw_id_card(draw, w, h):
    box_x = random.randint(15, 30)
    box_y = random.randint(15, 30)
    box_w = random.randint(60, 80)
    box_h = random.randint(80, 110)
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], outline="black", width=2)

    text_x = box_x + box_w + random.randint(15, 25)
    y = box_y + random.randint(0, 15)
    for _ in range(random.randint(4, 6)):
        line_end = w - random.randint(10, 30)
        draw.line([text_x, y, line_end, y], fill="black", width=2)
        y += random.randint(14, 22)


def draw_form(draw, w, h):
    rows = random.randint(3, 5)
    cols = random.randint(2, 4)
    box_w = random.randint(60, 85)
    box_h = random.randint(35, 50)
    x_gap = random.randint(10, 20)
    y_gap = random.randint(10, 20)
    start_x = random.randint(15, 30)
    start_y = random.randint(15, 30)

    for row in range(rows):
        for col in range(cols):
            x0 = start_x + col * (box_w + x_gap)
            y0 = start_y + row * (box_h + y_gap)
            if x0 + box_w < w - 10 and y0 + box_h < h - 10:
                draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], outline="black", width=1)


def draw_prescription(draw, w, h):
    for _ in range(random.randint(4, 7)):
        x0, y0 = random.randint(20, 100), random.randint(20, h - 20)
        x1, y1 = x0 + random.randint(80, 200), y0 + random.randint(-15, 15)
        draw.line([x0, y0, x1, y1], fill="black", width=2)


def draw_generic_photo(draw, w, h):
    for _ in range(random.randint(3, 6)):
        x0, y0 = random.randint(0, w - 50), random.randint(0, h - 50)
        x1, y1 = x0 + random.randint(20, 60), y0 + random.randint(20, 60)
        draw.ellipse([x0, y0, x1, y1], outline="black", width=2)

DRAW_FUNCS = {
    "invoice": draw_invoice,
    "receipt": draw_receipt,
    "id_card": draw_id_card,
    "form": draw_form,
    "prescription": draw_prescription,
    "generic_photo": draw_generic_photo,
}


def main():
    for category in CATEGORIES:
        out_path = OUT_DIR / category
        out_path.mkdir(parents=True, exist_ok=True)

        for i in range(IMAGES_PER_CATEGORY):
            w, h = 300, 400
            img = Image.new("RGB", (w, h), color="white")
            draw = ImageDraw.Draw(img)
            DRAW_FUNCS[category](draw, w, h)
            img.save(out_path / f"{category}_{i:03d}.png")

        print(f"Generated {IMAGES_PER_CATEGORY} images for '{category}'")


if __name__ == "__main__":
    main()