import hashlib
from pathlib import Path

import cv2
import numpy as np

TRAIN_DIR = Path("data/segmentation_grouped/train/images")
VAL_DIR = Path("data/segmentation_grouped/validation/images")


def md5_hash(path):
    """Calculate exact file hash."""

    hasher = hashlib.md5()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(8192),
            b""
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def perceptual_hash(path):
    """
    Simple perceptual hash.

    Resize image to 16x16 grayscale and compare
    pixels against the average brightness.
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None

    image = cv2.resize(
        image,
        (16, 16)
    )

    average = image.mean()

    binary = (
        image > average
    ).astype(np.uint8)

    return binary.flatten()


def hamming_distance(hash1, hash2):

    return np.count_nonzero(
        hash1 != hash2
    )


def main():

    train_images = sorted(
        TRAIN_DIR.glob("*.jpg")
    )

    val_images = sorted(
        VAL_DIR.glob("*.jpg")
    )

    print(
        f"Train images: {len(train_images)}"
    )

    print(
        f"Validation images: {len(val_images)}"
    )

    # =========================================
    # EXACT DUPLICATES
    # =========================================

    print(
        "\nChecking exact duplicates..."
    )

    train_md5 = {}

    for path in train_images:

        train_md5[
            md5_hash(path)
        ] = path

    exact_duplicates = []

    for path in val_images:

        file_hash = md5_hash(path)

        if file_hash in train_md5:

            exact_duplicates.append(
                (
                    train_md5[file_hash],
                    path
                )
            )

    print(
        f"Exact train/validation duplicates: "
        f"{len(exact_duplicates)}"
    )

    for train_path, val_path in exact_duplicates[:20]:

        print(
            "EXACT:",
            train_path,
            "<->",
            val_path
        )

    # =========================================
    # PERCEPTUAL / NEAR DUPLICATES
    # =========================================

    print(
        "\nCalculating perceptual hashes..."
    )

    train_phashes = []

    for path in train_images:

        image_hash = perceptual_hash(path)

        if image_hash is not None:

            train_phashes.append(
                (
                    path,
                    image_hash
                )
            )

    val_phashes = []

    for path in val_images:

        image_hash = perceptual_hash(path)

        if image_hash is not None:

            val_phashes.append(
                (
                    path,
                    image_hash
                )
            )

    print(
        "\nChecking suspiciously similar images..."
    )

    suspicious_pairs = []

    # Lower distance = more visually similar

    THRESHOLD = 15

    for val_path, val_hash in val_phashes:

        best_distance = 999
        best_train = None

        for train_path, train_hash in train_phashes:

            distance = hamming_distance(
                val_hash,
                train_hash
            )

            if distance < best_distance:

                best_distance = distance
                best_train = train_path

        if best_distance <= THRESHOLD:

            suspicious_pairs.append(
                (
                    best_distance,
                    best_train,
                    val_path
                )
            )

    suspicious_pairs.sort(
        key=lambda x: x[0]
    )

    print(
        f"Suspicious near-duplicate pairs: "
        f"{len(suspicious_pairs)}"
    )

    print(
        "\nMost similar pairs:"
    )

    for (
        distance,
        train_path,
        val_path
    ) in suspicious_pairs[:30]:

        print()

        print(
            f"Distance: {distance}"
        )

        print(
            f"TRAIN: {train_path}"
        )

        print(
            f"VAL:   {val_path}"
        )

    # =========================================
    # SUMMARY
    # =========================================

    print(
        "\n=============================="
    )

    print(
        "DATA LEAKAGE AUDIT SUMMARY"
    )

    print(
        "=============================="
    )

    print(
        f"Exact duplicates: "
        f"{len(exact_duplicates)}"
    )

    print(
        f"Suspicious near duplicates: "
        f"{len(suspicious_pairs)}"
    )

    if (
        len(exact_duplicates) == 0
        and
        len(suspicious_pairs) == 0
    ):

        print(
            "\nNo obvious image-level leakage detected."
        )

    else:

        print(
            "\nPotential leakage detected."
        )

        print(
            "Inspect suspicious pairs before "
            "trusting validation metrics."
        )


if __name__ == "__main__":
    main()
