
# InspectIQ

AI-powered steel surface defect classification using transfer learning.

## Overview

InspectIQ uses a pretrained ResNet18 model fine-tuned on the NEU-DET
steel surface defect dataset to classify images into six defect categories.

## Features

- Upload steel surface images through a web interface.
- Predict one of six defect classes.
- Display class probabilities and prediction confidence.
- Export prediction results as JSON.
- Serve predictions through a FastAPI REST API.

## Tech Stack

- Python
- PyTorch and TorchVision
- FastAPI and Uvicorn
- HTML, CSS and JavaScript
- Scikit-learn and Matplotlib

## Defect Classes

- Crazing
- Inclusion
- Patches
- Pitted surface
- Rolled-in scale
- Scratches

## Model Evaluation

The current model achieved 95.28% accuracy on the
360-image validation split.

This is a validation result, not independent test-set performance.

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure `models/inspectiq_resnet18.pth` is available.
5. Start the API:

   ```bash
   python -m uvicorn app.main:app --reload
   ```

6. Open http://127.0.0.1:8000

## Dataset

NEU Surface Defect Database / NEU-DET.

The dataset is not included in this repository. Obtain it from its
original source and follow the applicable dataset license and terms.

## Limitations

This is a research prototype for image-level defect classification.
It does not currently localize defects or replace professional
industrial quality inspection.
