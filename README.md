# DeepVerify — Document Intelligence & Visual Fraud Detection using Deep Learning

**Academic prototype.** Built with public/synthetic document datasets only
(MIDV-500 — synthetic identity documents created specifically for research).
No real, sensitive, or government-issued IDs should be used with this project.
This is not a production KYC/identity-verification system.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

> Important: the saved results are research-prototype results on a small
> synthetic dataset. They are not evidence of production fraud-detection
> accuracy.

## 1. Problem

Identity-document verification pipelines (as used in fintech/KYC onboarding)
typically need to: recognize what kind of document was submitted, extract
its text and structured fields, and flag documents that look visually
tampered or anomalous before handing them to a human reviewer. This project
builds a small end-to-end version of that pipeline and evaluates each stage
properly, rather than wrapping a single API call.

```
Document Image → Preprocessing (OpenCV) → Document Type Classification (CNN, transfer learning)
              → OCR / Text Extraction → Field Extraction → Tamper/Anomaly Detection
              → Confidence + Structured Result
```

## 2. Dataset

**[MIDV-500](https://l3i-share.univ-lr.fr)** — 500 video clips of 50 types
of synthetic identity documents (passports, driver's licenses, ID cards)
from different countries, filmed under varied lighting/angle/background
conditions, with ground-truth document-corner annotations. It was built
specifically for document-recognition research, which makes it a legitimate
stand-in for a real KYC dataset without any privacy concerns.

- Classes: document type (up to 50, or a subset — see `--max-per-class` /
  folder selection in `scripts/prepare_midv500.py`)
- Splits: the default helper creates stratified 70/15/15 train/val/test
  splits. If multiple frames come from the same source document or video,
  use a group-aware split before reporting final metrics; random frame-level
  splitting can overestimate performance.
- A synthetic fallback generator (`scripts/make_dummy_data.py`) is included
  purely to smoke-test the pipeline before the real dataset is downloaded.

See `scripts/prepare_midv500.py` for exact download/preparation instructions.

## 3. Architecture

| Stage | Technique |
|---|---|
| Preprocessing | OpenCV: contour-based perspective correction, denoising (`fastNlMeansDenoisingColored`), CLAHE brightness/contrast normalization |
| Document classification | Transfer learning: ResNet18 / ResNet34 / EfficientNet-B0 (ImageNet-pretrained) with a custom classification head |
| OCR | PaddleOCR in the integrated pipeline; Tesseract is retained for OCR experiments |
| Field extraction | Regex-based extraction over OCR text (dates, document numbers); designed as a slot to swap in a Transformer document-understanding model (LayoutLM/Donut) — see §6 |
| Tamper / anomaly detection | Two complementary signals: (1) classical Error Level Analysis (ELA) via JPEG re-compression diffing, (2) a small convolutional autoencoder trained only on genuine documents — high reconstruction error flags an anomaly |

Model code: `src/model.py`. Full inference pipeline: `src/inference.py`.

## 4. Training

```bash
python src/train.py --epochs 15 --model resnet18 --pretrained --augment
```

- Train/val/test split, early stopping on validation macro-F1, LR scheduling
  (`ReduceLROnPlateau`), checkpointing best model only.
- Device auto-detection: CUDA → Apple **MPS** (used on MacBook Air M4) → CPU
  (`src/model.py::get_device`).
- Optional unsupervised autoencoder for anomaly detection:
  `python scripts/train_autoencoder.py --epochs 20`

## 5. Experiments

All experiments log to `results/run_<timestamp>.json` for direct comparison.
Suggested comparisons for the writeup / interview:

1. **Pretrained vs. from-scratch**: `--pretrained` vs `--no-pretrained`
2. **Augmentation on/off**: `--augment` vs default (no augmentation) — tests
   robustness to orientation, lighting, and image-quality variation
3. **Backbone comparison**: `--model resnet18` vs `--model efficientnet_b0`
4. **Learning rate sweep**: `--lr 0.01 / 0.001 / 0.0001`
5. **Frozen vs fine-tuned backbone**: `--freeze-backbone` (linear-probe) vs
   full fine-tuning

Record accuracy/F1 for each run in a small comparison table before writing
final numbers in this README or your resume.

## 6. Evaluation

```bash
python src/evaluate.py --checkpoint results/best_model.pt
```

Outputs on the **held-out test set** (never used for training or model
selection):
- Per-class and macro precision/recall/F1 → `results/classification_report.txt`
- Confusion matrix → `results/confusion_matrix.png`

> Numbers are intentionally not hard-coded in this README — run the
> evaluation yourself and paste in your actual measured results before
> using them anywhere (resume, interview, etc.).

## 7. Transformer component (stretch goal)

`src/inference.py::extract_fields` currently uses regex over OCR text. If
time permits, replace this with a Hugging Face **LayoutLM** or **Donut**
model fine-tuned for key-value field extraction on document images — the
function signature (`ocr_text -> dict`) is designed so this is a drop-in
swap without touching the rest of the pipeline.

## 8. Demo

```bash
streamlit run app.py
```

Upload a **public/synthetic** document image and see every stage: original
→ preprocessed → predicted document type + confidence → OCR text →
extracted fields → tamper/anomaly score.

## 9. Project structure

```
DeepVerify/
├── data/
│   ├── raw/            # raw MIDV-500 download (not committed)
│   └── processed/      # data/processed/<class>/*.jpg used for training
├── notebooks/
│   └── experiments.ipynb
├── scripts/
│   ├── prepare_midv500.py   # raw MIDV-500 -> data/processed
│   ├── make_dummy_data.py   # synthetic smoke-test data
│   └── train_autoencoder.py # unsupervised anomaly detector
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   └── inference.py
├── app.py
├── requirements.txt
├── README.md
└── results/
    ├── best_model.pt
    ├── confusion_matrix.png
    └── training_curves.png
```

## 10. Setup (VS Code, macOS)

```bash
cd DeepVerify
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Install PaddleOCR and its compatible Paddle runtime for the integrated app.
# Install Tesseract separately only for the OCR experiment scripts.
```

Open the folder in VS Code, select `venv` as the interpreter
(Cmd+Shift+P → "Python: Select Interpreter"), then either use the provided
`.vscode/launch.json` run configurations or the terminal commands above.

Quick smoke test before downloading the real dataset:
```bash
python scripts/make_dummy_data.py
python src/train.py --epochs 3
python src/evaluate.py
streamlit run app.py
```

## 11. Measured results

The current saved run reports 100% accuracy and macro-F1 on 36 held-out
classification images across four classes. This is encouraging but too small
to support a generalization claim, and the split must be verified as
document/video-group-aware. The saved segmentation validation run has mean
Dice 0.9914 and mean IoU 0.9833 across 60 images, with lower-scoring outliers.

OCR and field extraction are not summarized by a labeled accuracy metric yet.
Tamper detection is also heuristic and currently has no labeled precision,
recall, F1, ROC-AUC, or calibrated fraud probability.

## 12. Limitations

- MIDV-500 is synthetic/staged data (actors' own documents photographed for
  research), not real production KYC traffic — results won't transfer
  directly to a production accuracy claim.
- Tamper/anomaly detection here is heuristic (ELA + reconstruction error),
  not a supervised forgery classifier trained on labeled tampered/genuine
  pairs, since no such public labeled dataset of comparable quality exists.
- Field extraction is regex-based by default; layout understanding is only
  as good as OCR quality plus the (optional) Transformer stretch goal.
- Not evaluated for fairness/bias across document nationalities — a real
  KYC system would need this analysis before any deployment claim.

## 13. Example predictions

_Add 2–3 example screenshots or JSON outputs here after training, e.g. from
`python src/inference.py --image data/processed/<class>/<file>.jpg`._
