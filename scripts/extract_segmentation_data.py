from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

SOURCE = Path("data/midv_download/extracted")
OUTPUT = Path("data/segmentation")

TRAIN_IMAGES = OUTPUT / "train/images"
TRAIN_MASKS = OUTPUT / "train/masks"

VAL_IMAGES = OUTPUT / "validation/images"
VAL_MASKS = OUTPUT / "validation/masks"


def decode_image(data):
    if data.get("bytes") is not None:
        return Image.open(BytesIO(data["bytes"]))

    return Image.open(data["path"])


def extract_file(parquet_path, split, start_index=0):

    print(f"\nReading: {parquet_path.name}")

    df = pd.read_parquet(parquet_path)

    if split == "train":
        image_dir = TRAIN_IMAGES
        mask_dir = TRAIN_MASKS
    else:
        image_dir = VAL_IMAGES
        mask_dir = VAL_MASKS

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    index = start_index

    for _, row in df.iterrows():

        image = decode_image(row["pixel_values"]).convert("RGB")
        mask = decode_image(row["label"]).convert("L")

        filename = f"{index:05d}"

        image.save(
            image_dir / f"{filename}.jpg",
            quality=95
        )

        mask.save(
            mask_dir / f"{filename}.png"
        )

        index += 1

    print(f"Extracted {len(df)} samples.")

    return index


def main():

    train_files = sorted(
        SOURCE.glob("data__train-*.parquet")
    )

    validation_files = sorted(
        SOURCE.glob("data__validation-*.parquet")
    )

    train_index = 0

    for file in train_files:
        train_index = extract_file(
            file,
            "train",
            train_index
        )

    validation_index = 0

    for file in validation_files:
        validation_index = extract_file(
            file,
            "validation",
            validation_index
        )

    print("\n==========================")
    print("EXTRACTION COMPLETE")
    print("==========================")

    print(f"Training samples: {train_index}")
    print(f"Validation samples: {validation_index}")

    print("\nDataset saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()