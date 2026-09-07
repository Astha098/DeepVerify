"""
inference.py
------------
Ties every stage of the pipeline together for a single uploaded image:

    Image -> preprocess -> classify document type -> OCR -> field extraction
          -> tamper/anomaly score -> final result dict

This is what app.py (the Streamlit demo) calls. It is also runnable
standalone for quick testing:

    python src/inference.py --image path/to/sample.jpg --checkpoint results/best_model.pt
"""

import argparse
import re

import cv2
import numpy as np
import pytesseract
import torch
import torch.nn.functional as F

from dataset import build_transforms
from model import ConvAutoencoder, DocumentClassifier, get_device
from preprocessing import error_level_analysis, preprocess_pipeline

FIELD_PATTERNS = {
    "date_of_birth": r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b",
    "document_number": r"\b([A-Z0-9]{6,12})\b",
    "expiry_date": r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b",
}


class DeepVerifyPipeline:
    def __init__(self, checkpoint_path="results/best_model.pt",
                 autoencoder_path=None, device=None):
        self.device = device or get_device()

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.class_to_idx = ckpt["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        train_args = ckpt["args"]
        self.img_size = train_args["img_size"]

        self.classifier = DocumentClassifier(
            num_classes=len(self.class_to_idx),
            backbone=train_args["model"],
            pretrained=False,
        ).to(self.device)
        self.classifier.load_state_dict(ckpt["model_state"])
        self.classifier.eval()

        self.transform = build_transforms(self.img_size, augment=False)

        self.autoencoder = None
        if autoencoder_path:
            self.autoencoder = ConvAutoencoder().to(self.device)
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
            self.autoencoder.eval()

    # ---- stage 1+2: preprocessing ----
    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        return preprocess_pipeline(image_bgr)

    # ---- stage 3: classification ----
    def classify(self, processed_bgr: np.ndarray):
        rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        return self.idx_to_class[pred_idx], float(probs[pred_idx]), probs.cpu().tolist()

    # ---- stage 4: OCR ----
    def extract_text(self, processed_bgr: np.ndarray) -> str:
        gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 15)
        text = pytesseract.image_to_string(gray)
        return text.strip()

    # ---- stage 5: field extraction (regex; swap in LayoutLM/Donut for real layouts) ----
    def extract_fields(self, ocr_text: str) -> dict:
        fields = {}
        for field, pattern in FIELD_PATTERNS.items():
            match = re.search(pattern, ocr_text)
            fields[field] = match.group(1) if match else None
        return fields

    # ---- stage 6: tamper / anomaly detection ----
    def anomaly_score(self, processed_bgr: np.ndarray) -> dict:
        ela = error_level_analysis(processed_bgr)
        ela_score = float(np.mean(ela))  # higher = more suspicious edit-boundary energy

        recon_error = None
        if self.autoencoder is not None:
            rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (128, 128)) / 255.0
            tensor = torch.tensor(small, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                recon = self.autoencoder(tensor)
                recon_error = float(F.mse_loss(recon, tensor).item())

        return {"ela_mean": ela_score, "autoencoder_recon_error": recon_error}

    # ---- full pipeline ----
    def run(self, image_bgr: np.ndarray) -> dict:
        processed = self.preprocess(image_bgr)
        doc_type, confidence, all_probs = self.classify(processed)
        ocr_text = self.extract_text(processed)
        fields = self.extract_fields(ocr_text)
        anomaly = self.anomaly_score(processed)

        return {
            "document_type": doc_type,
            "classification_confidence": round(confidence, 4),
            "class_probabilities": {
                self.idx_to_class[i]: round(p, 4) for i, p in enumerate(all_probs)
            },
            "ocr_text": ocr_text,
            "extracted_fields": fields,
            "anomaly": anomaly,
            "processed_image": processed,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default="results/best_model.pt")
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    pipeline = DeepVerifyPipeline(checkpoint_path=args.checkpoint)
    result = pipeline.run(image)
    result.pop("processed_image", None)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
