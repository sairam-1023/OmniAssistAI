"""
Trains the Vision classifier: fine-tunes a pretrained ResNet18 to
classify document images into the 6 categories, with Albumentations
augmentation and MLflow tracking.

Usage:
    python3 -m modules.vision.train
"""

from xml.parsers.expat import model

import albumentations as A
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models

from modules.core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

CATEGORIES = ["form", "generic_photo", "id_card", "invoice", "prescription", "receipt"]  # alphabetical, matches ImageFolder
IMG_SIZE = 224  # ResNet18's expected input size


# Training transforms: augmentation + normalization.
# Applied fresh, randomly, every time an image is loaded during training.
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Rotate(limit=8, p=0.5),                          # slight skew, like a crooked scan
    A.RandomBrightnessContrast(p=0.3),                  # uneven lighting/scanner exposure
    A.GaussNoise(p=0.2),                                # sensor/compression noise
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Validation transforms: NO augmentation, only resize + normalize.
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class AlbumentationsDataset(Dataset):
    """
    Wraps torchvision's ImageFolder (which handles finding files and
    assigning class labels from folder names) but swaps in Albumentations
    for the actual image transforms, since Albumentations expects numpy
    arrays, not PIL Images.
    """
    def __init__(self, root_dir, transform):
        self.image_folder = datasets.ImageFolder(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        path, label = self.image_folder.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))
        augmented = self.transform(image=image)
        return augmented["image"], label


def build_model(num_classes: int) -> nn.Module:
    """
    Loads ResNet18 pretrained on ImageNet, then replaces its final
    classification layer to output our num_classes instead of
    ImageNet's original 1000.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all pretrained layers — we don't want to destroy the
    # generic edge/texture features they already learned by training
    # them on our tiny 576-image dataset.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully-connected layer. This new layer is
    # created fresh (random weights) and IS trainable — it's the only
    # part of the network that actually learns our 6 categories.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()  # switch to training mode (affects layers like dropout/batchnorm, if present)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()          # clear gradients from the previous batch
        outputs = model(images)        # forward pass: model's raw predictions
        loss = criterion(outputs, labels)  # how wrong were we?
        loss.backward()                # backward pass: compute gradients
        optimizer.step()               # update the (unfrozen) weights using those gradients

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()  # switch to evaluation mode
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():  # don't compute/store gradients — we're not training here, saves memory
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    mlflow.set_experiment("omniassist-vision")

    train_ds = AlbumentationsDataset("data/images_split/train", train_transform)
    val_ds = AlbumentationsDataset("data/images_split/val", val_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = build_model(num_classes=len(CATEGORIES))
    criterion = nn.CrossEntropyLoss()
    # Only pass trainable params (our new fc layer) to the optimizer —
    # technically unnecessary since frozen params have requires_grad=False,
    # but explicit and clear about intent.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    num_epochs = 15

    with mlflow.start_run(run_name="vision-resnet18"):
        mlflow.log_params({
            "model": "resnet18",
            "img_size": IMG_SIZE,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "num_epochs": num_epochs,
            "frozen_backbone": True,
        })

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}"
            )
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            }, step=epoch)

        mlflow.pytorch.log_model(
        model.cpu(),
        name="vision_model",
        serialization_format="pickle",
        )
        model.to(DEVICE)  # move back to MPS in case we use `model` again after this point

    torch.save(model.state_dict(), "models/vision/resnet18_finetuned.pt")
    logger.info("Saved model to models/vision/resnet18_finetuned.pt")


if __name__ == "__main__":
    main()