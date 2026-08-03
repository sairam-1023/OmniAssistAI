"""
Evaluates the fine-tuned Vision model on real, held-back images ONLY
(data/real_images_split/eval/) — never seen during training in any form.
This is the honest measure of real-world performance.
"""

import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from modules.core.logging_config import configure_logging, get_logger
from modules.vision.train import AlbumentationsDataset, CATEGORIES, DEVICE, build_model, val_transform

configure_logging()
logger = get_logger(__name__)


def main():
    eval_ds = AlbumentationsDataset("data/real_images_split/eval", val_transform)
    eval_loader = DataLoader(eval_ds, batch_size=8, shuffle=False)

    model = build_model(num_classes=len(CATEGORIES))
    model.load_state_dict(torch.load("models/vision/resnet18_finetuned.pt", map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CATEGORIES)))
    logger.info(f"REAL-IMAGE confusion matrix (rows=actual, cols=predicted):\n{CATEGORIES}\n{cm}")
    logger.info(f"REAL-IMAGE classification report:\n{classification_report(all_labels, all_preds, target_names=CATEGORIES, labels=range(len(CATEGORIES)), zero_division=0)}")


if __name__ == "__main__":
    main()