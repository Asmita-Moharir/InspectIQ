
from pathlib import Path
from collections import Counter
from PIL import Image

DATA_DIR = Path("data/raw/NEU-DET")

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR.resolve()}\n"
        "Extract the downloaded dataset into this folder first."
    )

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
images = []
for path in DATA_DIR.rglob("*"):
    if path.is_file() and path.suffix.lower() in extensions:
        images.append(path)

print(f"Dataset directory: {DATA_DIR.resolve()}")
print(f"Total images found: {len(images)}")

if not images:
    print("No images found. Check the extracted directory structure.")
else:
    counts = Counter(
        str(image.parent.relative_to(DATA_DIR))
        for image in images
    )

    print("\nImages by folder:")
    for folder, count in sorted(counts.items()):
        print(f"{folder}: {count}")

    print("\nSample images:")
    for path in images[:5]:
        try:
            with Image.open(path) as image:
                print(
                    f"{path.relative_to(DATA_DIR)} | "
                    f"size={image.size} | mode={image.mode}"
                )
        except Exception as exc:
            print(f"Could not read {path}: {exc}")
