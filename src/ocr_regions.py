from pathlib import Path

import cv2
import pytesseract

# -------------------------------------------------
# Configuration
# -------------------------------------------------

IMAGE_PATH = Path(
    "results/document_extraction/extracted_document.jpg"
)

OUTPUT_DIR = Path(
    "results/ocr_regions"
)

UPSCALE = 4


# -------------------------------------------------
# Region definitions
#
# Coordinates are RELATIVE:
# (x1, y1, x2, y2)
#
# Values range from 0.0 to 1.0.
#
# These are approximate regions for the current
# ID-card layout and may need small adjustments.
# -------------------------------------------------

REGIONS = {

    # Main text block in center-left
    "identity_block": (
        0.28,
        0.10,
        0.72,
        0.68
    ),

    # Name area
    "name": (
        0.28,
        0.12,
        0.62,
        0.32
    ),

    # Nationality / birthplace
    "nationality_birthplace": (
        0.28,
        0.28,
        0.66,
        0.52
    ),

    # Date of birth
    "date_of_birth": (
        0.28,
        0.47,
        0.62,
        0.62
    ),

    # Right side document/card number
    "document_number": (
        0.66,
        0.18,
        0.98,
        0.42
    ),

    # Right-side dates
    "right_dates": (
        0.65,
        0.45,
        0.98,
        0.72
    ),

    # Personal number near lower right
    "personal_number": (
        0.63,
        0.62,
        0.98,
        0.83
    ),

    # Broad lower region, useful as backup
    "lower_text": (
        0.25,
        0.50,
        0.98,
        0.88
    ),
}


# -------------------------------------------------
# Crop using relative coordinates
# -------------------------------------------------

def crop_region(image, coordinates):

    height, width = image.shape[:2]

    x1, y1, x2, y2 = coordinates

    x1 = int(x1 * width)
    y1 = int(y1 * height)

    x2 = int(x2 * width)
    y2 = int(y2 * height)

    crop = image[
        y1:y2,
        x1:x2
    ]

    return crop


# -------------------------------------------------
# OCR preprocessing
# -------------------------------------------------

def preprocess_region(crop):

    # Upscale first

    upscaled = cv2.resize(
        crop,
        None,
        fx=UPSCALE,
        fy=UPSCALE,
        interpolation=cv2.INTER_CUBIC
    )

    # Convert to grayscale

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )

    # Mild denoising

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=5,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # CLAHE contrast enhancement

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        denoised
    )

    # Mild sharpening

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        blurred,
        -0.5,
        0
    )

    return sharpened


# -------------------------------------------------
# OCR
# -------------------------------------------------

def run_ocr(image, psm=6):

    config = (
        f"--oem 3 "
        f"--psm {psm} "
        f"-l eng"
    )

    text = pytesseract.image_to_string(
        image,
        config=config
    )

    return text.strip()


# -------------------------------------------------
# Numeric OCR
# -------------------------------------------------

def run_numeric_ocr(image):

    config = (
        "--oem 3 "
        "--psm 6 "
        "-l eng "
        "-c tessedit_char_whitelist="
        "0123456789-./"
    )

    text = pytesseract.image_to_string(
        image,
        config=config
    )

    return text.strip()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("REGION-BASED OCR")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load:\n{IMAGE_PATH}"
        )

    print(
        "\nImage size:",
        image.shape
    )

    all_results = []

    # Fields where numbers/dates are important

    numeric_regions = {
        "date_of_birth",
        "document_number",
        "right_dates",
        "personal_number"
    }

    for region_name, coordinates in REGIONS.items():

        print(
            "\n" + "=" * 60
        )

        print(
            "REGION:",
            region_name.upper()
        )

        print(
            "=" * 60
        )

        # Crop

        crop = crop_region(
            image,
            coordinates
        )

        if crop.size == 0:

            print(
                "Invalid/empty crop."
            )

            continue

        # Save original crop

        raw_path = (
            OUTPUT_DIR /
            f"{region_name}_raw.jpg"
        )

        cv2.imwrite(
            str(raw_path),
            crop
        )

        # Preprocess

        processed = preprocess_region(
            crop
        )

        processed_path = (
            OUTPUT_DIR /
            f"{region_name}_processed.png"
        )

        cv2.imwrite(
            str(processed_path),
            processed
        )

        # General OCR

        text_psm6 = run_ocr(
            processed,
            psm=6
        )

        text_psm11 = run_ocr(
            processed,
            psm=11
        )

        print(
            "\nPSM 6:"
        )

        print(
            text_psm6
            if text_psm6
            else "[No text]"
        )

        print(
            "\nPSM 11:"
        )

        print(
            text_psm11
            if text_psm11
            else "[No text]"
        )

        result = (
            f"\n{'=' * 60}\n"
            f"REGION: {region_name}\n"
            f"{'=' * 60}\n\n"
            f"PSM 6:\n"
            f"{text_psm6}\n\n"
            f"PSM 11:\n"
            f"{text_psm11}\n"
        )

        # Extra numeric OCR

        if region_name in numeric_regions:

            numeric_text = run_numeric_ocr(
                processed
            )

            print(
                "\nNUMERIC OCR:"
            )

            print(
                numeric_text
                if numeric_text
                else "[No text]"
            )

            result += (
                "\nNUMERIC OCR:\n"
                f"{numeric_text}\n"
            )

        all_results.append(
            result
        )

    # Save combined results

    results_path = (
        OUTPUT_DIR /
        "region_ocr_results.txt"
    )

    results_path.write_text(
        "\n".join(all_results),
        encoding="utf-8"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "REGION OCR COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nResults:"
    )

    print(
        results_path
    )


if __name__ == "__main__":
    main()