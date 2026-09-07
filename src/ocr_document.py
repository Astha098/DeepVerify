from pathlib import Path

import cv2
import pytesseract

# -------------------------------------------------
# Configuration
# -------------------------------------------------

INPUT_DIR = Path("results/ocr_preprocessed")
OUTPUT_DIR = Path("results/ocr")

# Test the most useful preprocessing variants
IMAGES = {
    "upscaled": INPUT_DIR / "01_upscaled.jpg",
    "gray": INPUT_DIR / "02_gray.png",
    "contrast": INPUT_DIR / "03_contrast.png",
    "denoised": INPUT_DIR / "04_denoised.png",
    "sharpened": INPUT_DIR / "05_sharpened.png",
    "otsu": INPUT_DIR / "06_otsu.png",
    "adaptive": INPUT_DIR / "07_adaptive.png",
}

# Tesseract page segmentation modes
PSM_MODES = [3, 6, 11, 12]


def run_ocr(image, psm):

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


def main():

    print("=" * 60)
    print("OCR EXPERIMENT")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    for variant_name, image_path in IMAGES.items():

        if not image_path.exists():

            print(
                f"\nSkipping missing file: "
                f"{image_path}"
            )

            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"\nCould not read: "
                f"{image_path}"
            )

            continue

        print(
            f"\nTesting: {variant_name.upper()}"
        )

        print(
            "Image size:",
            image.shape
        )

        for psm in PSM_MODES:

            print(
                "\n" + "=" * 60
            )

            print(
                f"{variant_name.upper()} | PSM {psm}"
            )

            print(
                "=" * 60
            )

            try:

                text = run_ocr(
                    image,
                    psm
                )

            except Exception as error:

                text = (
                    f"OCR ERROR: {error}"
                )

            if text:

                print(text)

            else:

                print(
                    "[No text detected]"
                )

            result = (
                f"\n{'=' * 60}\n"
                f"{variant_name.upper()} | "
                f"PSM {psm}\n"
                f"{'=' * 60}\n\n"
                f"{text}\n"
            )

            all_results.append(
                result
            )

            # Save individual result

            individual_file = (
                OUTPUT_DIR /
                f"{variant_name}_psm{psm}.txt"
            )

            individual_file.write_text(
                text,
                encoding="utf-8"
            )

    # Save combined results

    combined_path = (
        OUTPUT_DIR /
        "all_ocr_results.txt"
    )

    combined_path.write_text(
        "\n".join(all_results),
        encoding="utf-8"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "OCR EXPERIMENT COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nCombined results:"
    )

    print(
        combined_path
    )


if __name__ == "__main__":
    main()