"""
prepare_midv500.py
-------------------
Turns a raw MIDV-500 download into the flat classification layout the rest
of the project expects:

    data/processed/<document_type>/<unique_id>.jpg

WHERE TO GET THE RAW DATA (do this manually first, network access is not
available from inside this generation environment):

  Option A (recommended, easiest): Kaggle mirror
      1. pip install kaggle  (already in requirements.txt)
      2. Set up your Kaggle API token (~/.kaggle/kaggle.json)
      3. kaggle datasets download -d <search "MIDV-500" on kaggle.com and
         use the dataset slug you find> -p data/raw --unzip

  Option B: official Smart Engines source
      http://l3i-share.univ-lr.fr  or  ftp://smartengines.com/midv-500/dataset/
      Download the per-document-type zip files into data/raw/ and unzip
      them there, e.g.:
          data/raw/01_alb_id/images/...
          data/raw/01_alb_id/ground_truth/...
          data/raw/02_aut_drvlic_new/images/...
          ...

This script is deliberately tolerant of small structural differences
between mirrors: it walks data/raw recursively, treats each top-level
document-type folder as a class, and for every image tries to find a
matching ground-truth JSON with a "quad" (4-point polygon) to crop/warp
the document with our own perspective-correction code. If no ground truth
is found it falls back to using the full frame.

Usage:
    python scripts/prepare_midv500.py --raw-dir data/raw --out-dir data/processed
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocessing import order_points, perspective_correct  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_quad_in_json(json_path: Path):
    """MIDV-500 ground truth JSONs vary slightly by mirror; try common keys."""
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return None

    for key in ("quad", "corners", "points"):
        if key in data:
            pts = np.array(data[key], dtype="float32")
            if pts.shape == (4, 2):
                return pts

    # some mirrors nest under a single top-level object key
    for v in data.values():
        if isinstance(v, dict):
            for key in ("quad", "corners", "points"):
                if key in v:
                    pts = np.array(v[key], dtype="float32")
                    if pts.shape == (4, 2):
                        return pts
    return None


def crop_with_quad(image, quad, target_size=(640, 400)):
    rect = order_points(quad)
    (tl, tr, br, bl) = rect
    maxWidth = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxHeight = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if maxWidth < 10 or maxHeight < 10:
        return cv2.resize(image, target_size)
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return cv2.resize(warped, target_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--max-per-class", type=int, default=400,
                     help="cap images per class to keep training fast on a laptop")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"No folders found under {raw_dir}. Download MIDV-500 first (see docstring).")
        return

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if not images:
            continue

        out_class_dir = out_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in images:
            if count >= args.max_per_class:
                break
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            gt_candidates = list(img_path.parent.parent.rglob(f"{img_path.stem}.json"))
            quad = find_quad_in_json(gt_candidates[0]) if gt_candidates else None

            processed = crop_with_quad(image, quad) if quad is not None else perspective_correct(image)

            out_path = out_class_dir / f"{class_name}_{count:05d}.jpg"
            cv2.imwrite(str(out_path), processed)
            count += 1

        print(f"{class_name}: wrote {count} images")

    print(f"Done. Processed dataset written to {out_dir}")


if __name__ == "__main__":
    main()
