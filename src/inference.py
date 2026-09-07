"""
DeepVerify Integrated Inference Pipeline

Input image
    -> document segmentation
    -> document extraction / perspective correction
    -> document classification
    -> PaddleOCR
    -> field extraction
    -> anomaly analysis
"""

import argparse
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from paddleocr import PaddleOCR
from PIL import Image
from torchvision import transforms

from dataset import build_transforms
from extract_document import (
    clean_mask,
    find_document_corners,
    perspective_transform,
)
from model import ConvAutoencoder, DocumentClassifier, get_device
from preprocessing import error_level_analysis, preprocess_pipeline
from segmentation_model import UNet


class DeepVerifyPipeline:

    def __init__(
        self,
        checkpoint_path="results/best_model.pt",
        segmentation_model_path=None,
        autoencoder_path=None,
        device=None,
    ):

        self.device = device or get_device()

        # ====================================================
        # CLASSIFICATION MODEL
        # ====================================================

        ckpt = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.class_to_idx = ckpt["class_to_idx"]

        self.idx_to_class = {
            v: k
            for k, v in self.class_to_idx.items()
        }

        train_args = ckpt["args"]

        self.img_size = train_args["img_size"]

        self.classifier = DocumentClassifier(
            num_classes=len(self.class_to_idx),
            backbone=train_args["model"],
            pretrained=False,
        ).to(self.device)

        self.classifier.load_state_dict(
            ckpt["model_state"]
        )

        self.classifier.eval()

        self.transform = build_transforms(
            self.img_size,
            augment=False
        )

        # ====================================================
        # SEGMENTATION MODEL
        # ====================================================

        self.segmentation_model = None

        if segmentation_model_path:

            print(
                "Loading segmentation model..."
            )

            self.segmentation_model = UNet().to(
                self.device
            )

            segmentation_state = torch.load(
                segmentation_model_path,
                map_location=self.device
            )

            self.segmentation_model.load_state_dict(
                segmentation_state
            )

            self.segmentation_model.eval()

            print(
                "Segmentation model loaded."
            )

        self.segmentation_transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256)
                ),
                transforms.ToTensor(),
            ]
        )

        # ====================================================
        # PADDLE OCR
        # ====================================================

        print(
            "Loading PaddleOCR..."
        )

        self.ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        print(
            "PaddleOCR loaded."
        )

        # ====================================================
        # AUTOENCODER
        # ====================================================

        self.autoencoder = None

        if autoencoder_path:

            self.autoencoder = (
                ConvAutoencoder()
                .to(self.device)
            )

            self.autoencoder.load_state_dict(
                torch.load(
                    autoencoder_path,
                    map_location=self.device
                )
            )

            self.autoencoder.eval()

    # ========================================================
    # DOCUMENT EXTRACTION
    # ========================================================

    def extract_document(self, image_bgr):

        if self.segmentation_model is None:

            return image_bgr, {
                "used": False,
                "success": False,
                "area_ratio": None,
            }

        rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            rgb
        )

        original_height, original_width = (
            image_bgr.shape[:2]
        )

        input_tensor = (
            self.segmentation_transform(
                pil_image
            )
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            logits = self.segmentation_model(
                input_tensor
            )

            probability = torch.sigmoid(
                logits
            )

            prediction = (
                probability > 0.5
            ).float()

        mask = (
            prediction
            .squeeze()
            .cpu()
            .numpy()
            * 255
        ).astype(
            np.uint8
        )

        mask = cv2.resize(
            mask,
            (
                original_width,
                original_height
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        mask = clean_mask(
            mask
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:

            print(
                "WARNING: Segmentation found no document."
            )

            return image_bgr, {
                "used": True,
                "success": False,
                "area_ratio": 0.0,
            }

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        contour_area = cv2.contourArea(
            largest_contour
        )

        image_area = (
            original_width
            * original_height
        )

        area_ratio = (
            contour_area
            / image_area
        )

        points = find_document_corners(
            largest_contour
        )

        if points is None:

            rotated_rect = cv2.minAreaRect(
                largest_contour
            )

            points = cv2.boxPoints(
                rotated_rect
            ).astype(
                np.float32
            )

        extracted = perspective_transform(
            image_bgr,
            points
        )

        if (
            extracted is None
            or extracted.size == 0
        ):

            return image_bgr, {
                "used": True,
                "success": False,
                "area_ratio": round(
                    area_ratio,
                    4
                ),
            }

        height, width = (
            extracted.shape[:2]
        )

        if height > width:

            extracted = cv2.rotate(
                extracted,
                cv2.ROTATE_90_CLOCKWISE
            )

        return extracted, {
            "used": True,
            "success": True,
            "area_ratio": round(
                area_ratio,
                4
            ),
        }

    # ========================================================
    # PREPROCESSING
    # ========================================================

    def preprocess(
        self,
        image_bgr
    ):

        return preprocess_pipeline(
            image_bgr
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def classify(
        self,
        processed_bgr
    ):

        rgb = cv2.cvtColor(
            processed_bgr,
            cv2.COLOR_BGR2RGB
        )

        tensor = (
            self.transform(rgb)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            logits = self.classifier(
                tensor
            )

            probs = F.softmax(
                logits,
                dim=1
            )[0]

        pred_idx = int(
            probs.argmax()
        )

        confidence = float(
            probs[pred_idx]
        )

        predicted_class = (
            self.idx_to_class[
                pred_idx
            ]
        )

        # Low-confidence predictions should not
        # be presented as reliable classifications.

        if confidence < 0.60:

            final_class = "uncertain"

        else:

            final_class = predicted_class

        return (
            final_class,
            predicted_class,
            confidence,
            probs.cpu().tolist(),
        )

    # ========================================================
    # PADDLE OCR
    # ========================================================

    def extract_text(
        self,
        document_bgr
    ):

        results = self.ocr.predict(
            document_bgr
        )

        lines = []

        for result in results:

            try:

                data = result.json

                if callable(data):
                    data = data()

            except Exception:

                try:
                    data = dict(
                        result
                    )

                except Exception:
                    continue

            if not isinstance(
                data,
                dict
            ):
                continue

            content = data.get(
                "res",
                data
            )

            if not isinstance(
                content,
                dict
            ):
                continue

            texts = content.get(
                "rec_texts",
                []
            )

            scores = content.get(
                "rec_scores",
                []
            )

            for index, text in enumerate(
                texts
            ):

                text = str(
                    text
                ).strip()

                if not text:
                    continue

                confidence = None

                if index < len(scores):

                    try:

                        confidence = float(
                            scores[index]
                        )

                    except Exception:
                        pass

                lines.append(
                    {
                        "text": text,
                        "confidence": confidence,
                    }
                )

        full_text = "\n".join(
            item["text"]
            for item in lines
        )

        return full_text, lines

    # ========================================================
    # FIELD EXTRACTION
    # ========================================================

    def extract_fields(self, ocr_lines):

        fields = {
            "date_of_birth": None,
            "document_number": None,
            "expiry_date": None,
        }

        texts = [
            item["text"].strip()
            for item in ocr_lines
            if item.get("text")
        ]

        full_text = "\n".join(texts)

        date_pattern = re.compile(
            r"\b\d{2}[./-]\d{2}[./-]\d{4}\b"
        )

        # Extract recognized dates in reading order.
        # Common ID layouts typically present DOB before issue/expiry.
        dates = date_pattern.findall(full_text)

        if dates:
            fields["date_of_birth"] = dates[0]

        if len(dates) >= 2:
            fields["expiry_date"] = dates[-1]

        # Document/card number candidates must contain enough digits,
        # preventing ordinary words such as nationality values from
        # being mistaken for document numbers.
        id_pattern = re.compile(
            r"\b[A-Z0-9]{6,14}\b",
            re.IGNORECASE
        )

        candidates = []

        for text in texts:
            for match in id_pattern.findall(text):
                if date_pattern.fullmatch(match):
                    continue

                digit_count = sum(
                    character.isdigit()
                    for character in match
                )

                if digit_count >= 5:
                    candidates.append(match)

        # Prefer a numeric/alphanumeric candidate near a card/document
        # number label, allowing for intervening OCR lines.
        for index, text in enumerate(texts):
            lower = text.lower()

            if (
                "card no" in lower
                or "document no" in lower
                or "document number" in lower
            ):
                nearby = texts[index + 1:index + 6]

                for candidate_text in nearby:
                    matches = id_pattern.findall(candidate_text)

                    for match in matches:
                        if date_pattern.fullmatch(match):
                            continue

                        digit_count = sum(
                            character.isdigit()
                            for character in match
                        )

                        if digit_count >= 5:
                            fields["document_number"] = match
                            break

                    if fields["document_number"]:
                        break

            if fields["document_number"]:
                break

       
        return fields

    # ========================================================
    # ANOMALY DETECTION
    # ========================================================

    def anomaly_score(
        self,
        processed_bgr
    ):

        ela = error_level_analysis(
            processed_bgr
        )

        ela_score = float(
            np.mean(ela)
        )

        recon_error = None

        if self.autoencoder is not None:

            rgb = cv2.cvtColor(
                processed_bgr,
                cv2.COLOR_BGR2RGB
            )

            small = (
                cv2.resize(
                    rgb,
                    (128, 128)
                )
                / 255.0
            )

            tensor = (
                torch.tensor(
                    small,
                    dtype=torch.float32
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():

                recon = self.autoencoder(
                    tensor
                )

                recon_error = float(
                    F.mse_loss(
                        recon,
                        tensor
                    ).item()
                )

        return {
            "ela_mean": ela_score,
            "autoencoder_recon_error":
                recon_error,
        }

    # ========================================================
    # FULL PIPELINE
    # ========================================================

    def run(
        self,
        image_bgr
    ):

        # 1. Extract document

        document, extraction_info = (
            self.extract_document(
                image_bgr
            )
        )

        # 2. Classification preprocessing

        processed = self.preprocess(
            document
        )

        # 3. Classification

        (
            document_type,
            raw_prediction,
            confidence,
            all_probs,
        ) = self.classify(
            processed
        )

        # 4. PaddleOCR
        #
        # OCR uses the extracted original document,
        # NOT the classifier-preprocessed image.

        ocr_text, ocr_lines = (
            self.extract_text(
                document
            )
        )

        # 5. Fields

        fields = self.extract_fields(
            ocr_lines
        )

        # 6. Anomaly

        anomaly = self.anomaly_score(
            document
        )

        return {

            "document_extraction":
                extraction_info,

            "document_type":
                document_type,

            "raw_classification":
                raw_prediction,

            "classification_confidence":
                round(
                    confidence,
                    4
                ),

            "class_probabilities": {

                self.idx_to_class[i]:
                    round(p, 4)

                for i, p
                in enumerate(
                    all_probs
                )
            },

            "ocr_text":
                ocr_text,

            "ocr_lines":
                ocr_lines,

            "extracted_fields":
                fields,

            "anomaly":
                anomaly,

            "extracted_document":
                document,

            "processed_image":
                processed,
        }


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        default="results/best_model.pt"
    )

    parser.add_argument(
        "--segmentation-model",
        default=None
    )

    parser.add_argument(
        "--autoencoder",
        default=None
    )

    args = parser.parse_args()

    image = cv2.imread(
        args.image
    )

    if image is None:

        raise FileNotFoundError(
            args.image
        )

    pipeline = DeepVerifyPipeline(

        checkpoint_path=
            args.checkpoint,

        segmentation_model_path=
            args.segmentation_model,

        autoencoder_path=
            args.autoencoder,
    )

    result = pipeline.run(
        image
    )

    # Remove NumPy images before JSON printing.

    result.pop(
        "processed_image",
        None
    )

    result.pop(
        "extracted_document",
        None
    )

    import json

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":

    main()