import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

# Make src/ importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import DeepVerifyPipeline  # noqa: E402

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(
        description="Batch test DeepVerify on unseen documents."
    )

    parser.add_argument(
        "--input",
        default="data/unseen_test"
    )

    parser.add_argument(
        "--checkpoint",
        default="results/best_model.pt"
    )

    parser.add_argument(
        "--output",
        default="results/unseen_test"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        print(f"No images found in {input_dir}")
        return

    print("=" * 60)
    print("DEEPVERIFY — UNSEEN DOCUMENT BATCH TEST")
    print("=" * 60)

    print(f"\nImages found: {len(images)}")
    print("Loading DeepVerify model...")

    # Load model only once
    pipeline = DeepVerifyPipeline(
        checkpoint_path=args.checkpoint
    )

    print("Model loaded.\n")

    rows = []
    details = {}

    for i, image_path in enumerate(images, 1):

        print("-" * 60)
        print(f"[{i}/{len(images)}] {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:
            print("ERROR: Could not read image.")

            rows.append({
                "filename": image_path.name,
                "status": "ERROR",
                "document_type": "",
                "confidence": "",
                "ocr_characters": 0,
                "fields_found": 0,
                "ela_mean": "",
                "error": "Could not read image"
            })

            continue

        try:
            result = pipeline.run(image)

            processed = result.pop(
                "processed_image",
                None
            )

            doc_type = result.get(
                "document_type"
            )

            confidence = result.get(
                "classification_confidence"
            )

            ocr_text = result.get(
                "ocr_text", ""
            )

            fields = result.get(
                "extracted_fields", {}
            )

            fields_found = sum(
                value is not None
                for value in fields.values()
            )

            anomaly = result.get(
                "anomaly", {}
            )

            ela_mean = anomaly.get(
                "ela_mean"
            )

            print(f"Prediction     : {doc_type}")
            print(f"Confidence     : {confidence}")
            print(f"OCR characters: {len(ocr_text)}")
            print(f"Fields found   : {fields_found}/{len(fields)}")
            print(f"ELA mean       : {ela_mean}")
            print("Pipeline status: SUCCESS")

            # Save processed image
            if processed is not None:
                processed_path = (
                    output_dir /
                    f"{image_path.stem}_processed.jpg"
                )

                cv2.imwrite(
                    str(processed_path),
                    processed
                )

            rows.append({
                "filename": image_path.name,
                "status": "SUCCESS",
                "document_type": doc_type,
                "confidence": confidence,
                "ocr_characters": len(ocr_text),
                "fields_found": fields_found,
                "ela_mean": ela_mean,
                "error": ""
            })

            details[image_path.name] = result

        except Exception as e:

            print(f"ERROR: {e}")

            rows.append({
                "filename": image_path.name,
                "status": "ERROR",
                "document_type": "",
                "confidence": "",
                "ocr_characters": 0,
                "fields_found": 0,
                "ela_mean": "",
                "error": str(e)
            })

    # Save CSV
    csv_path = output_dir / "batch_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "status",
                "document_type",
                "confidence",
                "ocr_characters",
                "fields_found",
                "ela_mean",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # Save full JSON results
    json_path = output_dir / "detailed_results.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            details,
            f,
            indent=2,
            ensure_ascii=False
        )

    successful = sum(
        row["status"] == "SUCCESS"
        for row in rows
    )

    failed = len(rows) - successful

    print("\n" + "=" * 60)
    print("BATCH TEST COMPLETE")
    print("=" * 60)

    print(f"Total images : {len(rows)}")
    print(f"Processed    : {successful}")
    print(f"Errors       : {failed}")

    print(f"\nCSV  : {csv_path}")
    print(f"JSON : {json_path}")


if __name__ == "__main__":
    main()
