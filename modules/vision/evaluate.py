"""
Loads the trained Vision model and generates a confusion matrix on the
validation set, so we can see exactly which categories (if any) get
confused with each other — not just a single aggregate accuracy number.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from modules.core.logging_config import configure_logging, get_logger
from modules.vision.train import AlbumentationsDataset, CATEGORIES, DEVICE, build_model, val_transform

configure_logging()
logger = get_logger(__name__)


def main():
    val_ds = AlbumentationsDataset("data/images_split/val", val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = build_model(num_classes=len(CATEGORIES))
    model.load_state_dict(torch.load("models/vision/resnet18_finetuned.pt", map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"Confusion matrix (rows=actual, cols=predicted):\n{CATEGORIES}\n{cm}")
    logger.info(f"Classification report:\n{classification_report(all_labels, all_preds, target_names=CATEGORIES)}")


if __name__ == "__main__":
    main()