
from pathlib import Path

import io

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "inspectiq_resnet18.pth"
TEMPLATE_PATH = ROOT / "app" / "templates" / "index.html"
STATIC_DIR = ROOT / "app" / "static"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

app = FastAPI(
    title="InspectIQ API",
    description="Steel surface defect classification API",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Trained model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        classes = checkpoint.get("class_names", CLASSES)
    else:
        state_dict = checkpoint
        classes = CLASSES

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, classes


model, class_names = load_model()


@app.get("/")
def home():
    if not TEMPLATE_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend HTML file not found.",
        )

    return FileResponse(TEMPLATE_PATH)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "ResNet18",
        "device": str(DEVICE),
        "classes": class_names,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WebP image.",
        )

    # Limit uploads to 10 MB.
    content = await file.read(10 * 1024 * 1024 + 1)

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB.",
        )

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    order = torch.argsort(
        probabilities,
        descending=True,
    ).tolist()

    predictions = [
        {
            "class_name": class_names[index],
            "confidence": round(
                probabilities[index].item() * 100, 2
            ),
        }
        for index in order
    ]

    return {
        "filename": file.filename,
        "predicted_class": predictions[0]["class_name"],
        "confidence": predictions[0]["confidence"],
        "predictions": predictions,
    }
