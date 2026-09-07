from pathlib import Path

import cv2

# -------------------------------------------------
# Configuration
# -------------------------------------------------

INPUT_PATH = Path(
    "results/document_extraction/extracted_document.jpg"
)

OUTPUT_DIR = Path(
    "results/ocr_preprocessed"
)

# Upscale factor.
# Small document text benefits significantly from enlargement.
UPSCALE_FACTOR = 3


# -------------------------------------------------
# Resize / Upscale
# -------------------------------------------------

def upscale_image(image, scale=3):

    height, width = image.shape[:2]

    new_width = width * scale
    new_height = height * scale

    upscaled = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )

    return upscaled


# -------------------------------------------------
# CLAHE Contrast Enhancement
# -------------------------------------------------

def enhance_contrast(gray):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


# -------------------------------------------------
# Denoising
# -------------------------------------------------

def denoise_image(gray):

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=7,
        templateWindowSize=7,
        searchWindowSize=21
    )

    return denoised


# -------------------------------------------------
# Sharpening
# -------------------------------------------------

def sharpen_image(gray):

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=2
    )

    sharpened = cv2.addWeighted(
        gray,
        1.8,
        blurred,
        -0.8,
        0
    )

    return sharpened


# -------------------------------------------------
# Otsu Threshold
# -------------------------------------------------

def otsu_threshold(gray):

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return threshold


# -------------------------------------------------
# Adaptive Threshold
# -------------------------------------------------

def adaptive_threshold(gray):

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return threshold


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("OCR PREPROCESSING")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------
    # Load extracted document
    # ---------------------------------------------

    image = cv2.imread(
        str(INPUT_PATH)
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image:\n{INPUT_PATH}"
        )

    print("\nInput image:")
    print(INPUT_PATH)

    print(
        "Original size:",
        image.shape
    )

    # ---------------------------------------------
    # 1. Upscale
    # ---------------------------------------------

    upscaled = upscale_image(
        image,
        UPSCALE_FACTOR
    )

    print(
        "Upscaled size:",
        upscaled.shape
    )

    # ---------------------------------------------
    # 2. Convert to grayscale
    # ---------------------------------------------

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------------------
    # 3. Contrast enhancement
    # ---------------------------------------------

    contrast = enhance_contrast(
        gray
    )

    # ---------------------------------------------
    # 4. Denoising
    # ---------------------------------------------

    denoised = denoise_image(
        contrast
    )

    # ---------------------------------------------
    # 5. Sharpen
    # ---------------------------------------------

    sharpened = sharpen_image(
        denoised
    )

    # ---------------------------------------------
    # 6. Threshold variants
    # ---------------------------------------------

    otsu = otsu_threshold(
        sharpened
    )

    adaptive = adaptive_threshold(
        sharpened
    )

    # ---------------------------------------------
    # Save all preprocessing variants
    # ---------------------------------------------

    outputs = {

        "01_upscaled.jpg":
            upscaled,

        "02_gray.png":
            gray,

        "03_contrast.png":
            contrast,

        "04_denoised.png":
            denoised,

        "05_sharpened.png":
            sharpened,

        "06_otsu.png":
            otsu,

        "07_adaptive.png":
            adaptive,
    }

    print("\nSaving preprocessing variants:")

    for filename, processed_image in outputs.items():

        output_path = (
            OUTPUT_DIR /
            filename
        )

        success = cv2.imwrite(
            str(output_path),
            processed_image
        )

        if success:

            print(
                "Saved:",
                output_path
            )

        else:

            print(
                "FAILED:",
                output_path
            )

    print("\n" + "=" * 60)

    print(
        "OCR PREPROCESSING COMPLETE"
    )

    print("=" * 60)

    print(
        "\nResults saved in:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()