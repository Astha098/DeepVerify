import json
from pathlib import Path

from paddleocr import PaddleOCR

# -------------------------------------------------
# Configuration
# -------------------------------------------------

IMAGE_PATH = Path(
    "results/document_extraction/extracted_document.jpg"
)

OUTPUT_DIR = Path(
    "results/paddle_ocr"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("PADDLEOCR DOCUMENT RECOGNITION")
    print("=" * 60)

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"Image not found:\n{IMAGE_PATH}"
        )

    print("\nInput:")
    print(IMAGE_PATH)

    # ---------------------------------------------
    # Initialize PaddleOCR
    #
    # PaddleOCR 3.x API
    # ---------------------------------------------

    print("\nLoading PaddleOCR model...")

    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    print("Model loaded.")

    # ---------------------------------------------
    # Run OCR
    # ---------------------------------------------

    print("\nRunning OCR...")

    results = ocr.predict(
        str(IMAGE_PATH)
    )

    print("\nOCR finished.")

    # ---------------------------------------------
    # Collect text
    # ---------------------------------------------

    extracted_lines = []

    raw_results = []

    for result in results:

        # PaddleOCR 3.x result behaves like
        # a dictionary/result object.

        try:

            data = result.json

            if callable(data):
                data = data()

        except Exception:

            try:
                data = dict(result)

            except Exception:
                data = {
                    "raw": str(result)
                }

        raw_results.append(
            data
        )

        # Try to extract recognized text

        if isinstance(data, dict):

            # Some versions wrap data under "res"

            content = data.get(
                "res",
                data
            )

            if isinstance(
                content,
                dict
            ):

                texts = content.get(
                    "rec_texts",
                    []
                )

                scores = content.get(
                    "rec_scores",
                    []
                )

                for index, text in enumerate(texts):

                    text = str(
                        text
                    ).strip()

                    if not text:
                        continue

                    score = None

                    if index < len(scores):

                        try:
                            score = float(
                                scores[index]
                            )

                        except Exception:
                            pass

                    extracted_lines.append({
                        "text": text,
                        "confidence": score
                    })

    # ---------------------------------------------
    # Print recognized text
    # ---------------------------------------------

    print(
        "\n" +
        "=" * 60
    )

    print(
        "RECOGNIZED TEXT"
    )

    print(
        "=" * 60
    )

    if not extracted_lines:

        print(
            "\nNo structured text was extracted."
        )

        print(
            "Raw PaddleOCR output will still "
            "be saved for inspection."
        )

    else:

        for item in extracted_lines:

            text = item[
                "text"
            ]

            confidence = item[
                "confidence"
            ]

            if confidence is not None:

                print(
                    f"{text} "
                    f"[confidence: "
                    f"{confidence:.3f}]"
                )

            else:

                print(
                    text
                )

    # ---------------------------------------------
    # Save clean text
    # ---------------------------------------------

    text_path = (
        OUTPUT_DIR /
        "recognized_text.txt"
    )

    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as file:

        for item in extracted_lines:

            confidence = item[
                "confidence"
            ]

            if confidence is None:

                file.write(
                    item["text"] +
                    "\n"
                )

            else:

                file.write(
                    f'{item["text"]}'
                    f'\t'
                    f'{confidence:.4f}'
                    f'\n'
                )

    # ---------------------------------------------
    # Save structured OCR output
    # ---------------------------------------------

    structured_path = (
        OUTPUT_DIR /
        "structured_ocr.json"
    )

    with open(
        structured_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            extracted_lines,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ---------------------------------------------
    # Save raw result
    # ---------------------------------------------

    raw_path = (
        OUTPUT_DIR /
        "raw_ocr_output.txt"
    )

    with open(
        raw_path,
        "w",
        encoding="utf-8"
    ) as file:

        for result in raw_results:

            file.write(
                str(result)
            )

            file.write(
                "\n\n"
            )

    print(
        "\n" +
        "=" * 60
    )

    print(
        "PADDLEOCR COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nResults saved to:"
    )

    print(
        text_path
    )

    print(
        structured_path
    )

    print(
        raw_path
    )


if __name__ == "__main__":
    main()