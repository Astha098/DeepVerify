"""
make_dummy_data.py
-------------------
Creates a tiny synthetic dataset under data/processed/ so you can verify the
whole pipeline (dataset -> model -> train -> evaluate -> inference -> app)
runs end-to-end on your machine BEFORE spending time downloading the real
MIDV-500 data. Not meant for real training/experiments -- just a smoke test.

Usage:
    python scripts/make_dummy_data.py
"""

from pathlib import Path

import cv2
import numpy as np

CLASSES = ["passport", "drivers_license", "national_id", "residence_permit"]
N_PER_CLASS = 60
OUT_DIR = Path("data/processed")


def make_fake_document(class_idx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((400, 640, 3), 230, dtype=np.uint8)
    # class-dependent color band so the classifier has a real (if trivial) signal to learn
    color = [(60, 90, 200), (40, 160, 60), (200, 120, 40), (150, 60, 160)][class_idx % 4]
    cv2.rectangle(img, (0, 0), (640, 60), color, -1)
    cv2.putText(img, f"SAMPLE DOCUMENT {class_idx}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    for i in range(6):
        y = 100 + i * 40
        text = "".join(chr(rng.integers(65, 90)) for _ in range(rng.integers(8, 20)))
        cv2.putText(img, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    noise = rng.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def main():
    for idx, cls in enumerate(CLASSES):
        cls_dir = OUT_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(N_PER_CLASS):
            img = make_fake_document(idx, seed=idx * 1000 + i)
            cv2.imwrite(str(cls_dir / f"{cls}_{i:04d}.jpg"), img)
        print(f"{cls}: wrote {N_PER_CLASS} synthetic images")
    print(f"Dummy dataset ready at {OUT_DIR}. Now try: python src/train.py --epochs 3")


if __name__ == "__main__":
    main()
