import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from segmentation_model import UNet

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path(
    "data/segmentation_grouped/validation/images"
)

MASK_DIR = Path(
    "data/segmentation_grouped/validation/masks"
)

MODEL_PATH = Path(
    "results/best_segmentation_model_grouped.pth"
)

OUTPUT_DIR = Path(
    "results/grouped_predictions"
)

REPORT_PATH = Path(
    "results/grouped_validation_metrics.csv"
)

IMAGE_SIZE = 256


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target):

    prediction = prediction.astype(
        np.float32
    )

    target = target.astype(
        np.float32
    )

    intersection = (
        prediction * target
    ).sum()

    dice = (
        2.0 * intersection + 1e-7
    ) / (
        prediction.sum()
        + target.sum()
        + 1e-7
    )

    union = (
        prediction
        + target
        - prediction * target
    ).sum()

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return float(dice), float(iou)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    original,
    true_mask,
    prediction,
    image_name,
    dice,
    iou,
    category
):

    category_dir = (
        OUTPUT_DIR /
        category
    )

    category_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig = plt.figure(
        figsize=(12, 4)
    )

    # Original

    ax1 = fig.add_subplot(
        1, 3, 1
    )

    ax1.imshow(
        original.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    )

    ax1.set_title(
        "Original Image"
    )

    ax1.axis(
        "off"
    )

    # Ground truth

    ax2 = fig.add_subplot(
        1, 3, 2
    )

    ax2.imshow(
        true_mask,
        cmap="gray"
    )

    ax2.set_title(
        "Ground Truth"
    )

    ax2.axis(
        "off"
    )

    # Prediction

    ax3 = fig.add_subplot(
        1, 3, 3
    )

    ax3.imshow(
        prediction,
        cmap="gray"
    )

    ax3.set_title(
        f"Prediction\n"
        f"Dice: {dice:.4f} | "
        f"IoU: {iou:.4f}"
    )

    ax3.axis(
        "off"
    )

    plt.tight_layout()

    output_path = (
        category_dir /
        f"{image_name}_prediction.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved {category}:",
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    device = get_device()

    print(
        "Using device:",
        device
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Validation dataset:",
        IMAGE_DIR
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load best grouped model
    # --------------------------------------------------------

    model = UNet().to(
        device
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    print(
        "\nBest model loaded successfully."
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor()

    ])

    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    print(
        f"\nValidation images found: "
        f"{len(image_paths)}"
    )

    results = []

    # ========================================================
    # RUN INFERENCE ON ALL VALIDATION IMAGES
    # ========================================================

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        mask_path = (
            MASK_DIR /
            f"{image_path.stem}.png"
        )

        if not mask_path.exists():

            print(
                "Missing mask:",
                mask_path
            )

            continue

        # Original image

        original = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        # Ground truth

        true_mask_image = Image.open(
            mask_path
        ).convert(
            "L"
        )

        true_mask_image = (
            true_mask_image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                Image.Resampling.NEAREST
            )
        )

        true_mask = np.array(
            true_mask_image,
            dtype=np.float32
        )

        true_mask = (
            true_mask > 127
        ).astype(
            np.float32
        )

        # Prepare input

        input_tensor = (
            transform(
                original
            )
            .unsqueeze(0)
            .to(device)
        )

        # Prediction

        with torch.no_grad():

            logits = model(
                input_tensor
            )

            probability = torch.sigmoid(
                logits
            )

            prediction = (
                probability > 0.5
            ).float()

        prediction = (
            prediction
            .squeeze()
            .cpu()
            .numpy()
        )

        # Metrics

        dice, iou = (
            calculate_metrics(
                prediction,
                true_mask
            )
        )

        results.append({

            "image":
                image_path.name,

            "dice":
                dice,

            "iou":
                iou,

            "original":
                original,

            "true_mask":
                true_mask,

            "prediction":
                prediction

        })

        print(
            f"[{index:02d}/"
            f"{len(image_paths)}] "
            f"{image_path.name} "
            f"| Dice: {dice:.4f} "
            f"| IoU: {iou:.4f}"
        )

    if not results:

        print(
            "No predictions generated."
        )

        return

    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    dice_scores = [
        result["dice"]
        for result in results
    ]

    iou_scores = [
        result["iou"]
        for result in results
    ]

    mean_dice = np.mean(
        dice_scores
    )

    mean_iou = np.mean(
        iou_scores
    )

    median_dice = np.median(
        dice_scores
    )

    min_dice = np.min(
        dice_scores
    )

    max_dice = np.max(
        dice_scores
    )

    print(
        "\n"
        + "=" * 55
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 55
    )

    print(
        f"Images evaluated: "
        f"{len(results)}"
    )

    print(
        f"Mean Dice:   "
        f"{mean_dice:.4f}"
    )

    print(
        f"Mean IoU:    "
        f"{mean_iou:.4f}"
    )

    print(
        f"Median Dice: "
        f"{median_dice:.4f}"
    )

    print(
        f"Lowest Dice: "
        f"{min_dice:.4f}"
    )

    print(
        f"Highest Dice:"
        f" {max_dice:.4f}"
    )

    # ========================================================
    # SAVE CSV REPORT
    # ========================================================

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "image",
            "dice",
            "iou"
        ])

        for result in results:

            writer.writerow([

                result["image"],

                f'{result["dice"]:.6f}',

                f'{result["iou"]:.6f}'

            ])

    print(
        "\nMetrics saved:"
    )

    print(
        REPORT_PATH
    )

    # ========================================================
    # SORT BY PERFORMANCE
    # ========================================================

    results.sort(
        key=lambda x: x["dice"]
    )

    # Worst 5

    worst = results[:5]

    # Best 5

    best = results[-5:][::-1]

    # 5 around median

    middle_index = (
        len(results) // 2
    )

    start = max(
        0,
        middle_index - 2
    )

    middle = results[
        start:
        start + 5
    ]

    # ========================================================
    # SAVE REPRESENTATIVE EXAMPLES
    # ========================================================

    print(
        "\nSaving worst predictions..."
    )

    for result in worst:

        save_visualization(

            result["original"],

            result["true_mask"],

            result["prediction"],

            Path(
                result["image"]
            ).stem,

            result["dice"],

            result["iou"],

            "worst"

        )

    print(
        "\nSaving middle predictions..."
    )

    for result in middle:

        save_visualization(

            result["original"],

            result["true_mask"],

            result["prediction"],

            Path(
                result["image"]
            ).stem,

            result["dice"],

            result["iou"],

            "middle"

        )

    print(
        "\nSaving best predictions..."
    )

    for result in best:

        save_visualization(

            result["original"],

            result["true_mask"],

            result["prediction"],

            Path(
                result["image"]
            ).stem,

            result["dice"],

            result["iou"],

            "best"

        )

    print(
        "\n"
        + "=" * 55
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 55
    )

    print(
        "\nVisualizations:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()