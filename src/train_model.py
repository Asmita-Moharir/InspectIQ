
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# ---------------- CONFIGURATION ----------------
ROOT = Path("data/raw/NEU-DET")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_DIR = ROOT / "train" / "images"
VAL_DIR = ROOT / "validation" / "images"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(
            "Training or validation directory not found. "
            "Check data/raw/NEU-DET."
        )

    # Use CPU by default; CUDA is used automatically if available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ImageFolder obtains labels from the class subdirectories.
    train_dataset = datasets.ImageFolder(
        TRAIN_DIR, transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        VAL_DIR, transform=val_transform
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            f"Class mismatch: {train_dataset.classes} vs "
            f"{val_dataset.classes}"
        )

    classes = train_dataset.classes
    num_classes = len(classes)

    print(f"Classes: {classes}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # Windows-compatible data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Load pretrained ResNet-18
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze pretrained feature extractor
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace classification layer
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.fc.parameters(), lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0
    history = []

    for epoch in range(EPOCHS):
        # -------- TRAINING --------
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_loss = running_loss / train_total
        train_accuracy = train_correct / train_total

        # -------- VALIDATION --------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (
                    predictions == labels
                ).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_accuracy = val_correct / val_total

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2%}"
        )

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "input_size": 224,
                "architecture": "resnet18",
            }, MODEL_DIR / "inspectiq_resnet18.pth")

            print("  -> Best model saved.")

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=4)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Model saved to: {MODEL_DIR / 'inspectiq_resnet18.pth'}")
    print(f"History saved to: {OUTPUT_DIR / 'training_history.json'}")


if __name__ == "__main__":
    main()
