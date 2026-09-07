from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# ============================================================
# REMAINING SUSPICIOUS PAIRS FROM GROUP-AWARE SPLIT
# ============================================================

PAIRS = [
    ("00086.jpg", "00035.jpg", 11),
    ("00086.jpg", "00036.jpg", 11),
    ("00153.jpg", "00029.jpg", 12),
    ("00179.jpg", "00034.jpg", 13),
    ("00187.jpg", "00001.jpg", 14),
]


# ============================================================
# USE THE NEW GROUP-AWARE DATASET
# ============================================================

TRAIN_DIR = Path(
    "data/segmentation_grouped/train/images"
)

VAL_DIR = Path(
    "data/segmentation_grouped/validation/images"
)


# Save separately so the old comparison is not overwritten
OUTPUT = Path(
    "results/grouped_leakage_comparison.png"
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_rgb(path):

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not find image: {path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Checking {len(PAIRS)} suspicious pairs..."
    )

    # Create comparison figure
    fig, axes = plt.subplots(
        len(PAIRS),
        2,
        figsize=(
            10,
            4 * len(PAIRS)
        )
    )

    for i, (
        train_name,
        val_name,
        distance
    ) in enumerate(PAIRS):

        train_path = (
            TRAIN_DIR /
            train_name
        )

        val_path = (
            VAL_DIR /
            val_name
        )

        print()
        print(
            f"Pair {i + 1}"
        )

        print(
            f"TRAIN: {train_path}"
        )

        print(
            f"VAL:   {val_path}"
        )

        print(
            f"Hash distance: {distance}"
        )

        # Load images
        train_image = load_rgb(
            train_path
        )

        val_image = load_rgb(
            val_path
        )

        # ------------------------------------------
        # TRAIN IMAGE
        # ------------------------------------------

        axes[i, 0].imshow(
            train_image
        )

        axes[i, 0].set_title(
            f"TRAIN\n"
            f"{train_name}"
        )

        axes[i, 0].axis(
            "off"
        )

        # ------------------------------------------
        # VALIDATION IMAGE
        # ------------------------------------------

        axes[i, 1].imshow(
            val_image
        )

        axes[i, 1].set_title(
            f"VALIDATION\n"
            f"{val_name}\n"
            f"Hash Distance: {distance}"
        )

        axes[i, 1].axis(
            "off"
        )

    # Main title
    fig.suptitle(
        "Group-Aware Split — Remaining Suspicious Pairs",
        fontsize=16
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    # Save result
    plt.savefig(
        OUTPUT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "=" * 50
    )

    print(
        "VISUAL LEAKAGE CHECK COMPLETE"
    )

    print(
        "=" * 50
    )

    print(
        "Saved comparison to:"
    )

    print(
        OUTPUT
    )


if __name__ == "__main__":
    main()