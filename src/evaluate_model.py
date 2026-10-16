
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "data" / "raw" / "NEU-DET" / "validation" / "images"
MODEL_PATH = ROOT / "models" / "inspectiq_resnet18.pth"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

# Support checkpoints saved as either a dictionary or a state dict.
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    class_names = checkpoint.get("class_names", dataset.classes)
else:
    state_dict = checkpoint
    class_names = dataset.classes

if class_names != dataset.classes:
    raise ValueError(
        f"Class mismatch: model={class_names}, data={dataset.classes}"
    )

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

all_true = []
all_pred = []

with torch.no_grad():
    for images, labels in loader:
        logits = model(images.to(DEVICE))
        predictions = logits.argmax(dim=1).cpu()

        all_true.extend(labels.tolist())
        all_pred.extend(predictions.tolist())

print("\nPer-class evaluation:\n")
print(classification_report(
    all_true,
    all_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0,
))

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=list(range(len(class_names))),
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
ax.set_title("InspectIQ - Validation Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

print(f"\nConfusion matrix saved to: {OUTPUT_DIR / 'confusion_matrix.png'}")
