import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from segmentation_model import UNet

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_MODEL_PATH = Path(
    "results/best_segmentation_model_grouped.pth"
)

DEFAULT_IMAGE_PATH = Path(
    "data/segmentation_grouped/validation/images/00029.jpg"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/document_extraction"
)

IMAGE_SIZE = 256
MASK_THRESHOLD = 0.5


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
# ORDER FOUR CORNER POINTS
# ============================================================

def order_points(points):

    """
    Order four points as:

    top-left
    top-right
    bottom-right
    bottom-left
    """

    points = np.asarray(
        points,
        dtype=np.float32
    ).reshape(4, 2)

    rect = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    s = points.sum(axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(
        points,
        axis=1
    ).reshape(-1)

    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


# ============================================================
# PERSPECTIVE TRANSFORMATION
# ============================================================

def perspective_transform(image, points):

    """
    Flatten the detected document using four corner points.
    """

    rect = order_points(points)

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(
        br - bl
    )

    width_b = np.linalg.norm(
        tr - tl
    )

    max_width = max(
        1,
        int(round(max(width_a, width_b)))
    )

    height_a = np.linalg.norm(
        tr - br
    )

    height_b = np.linalg.norm(
        tl - bl
    )

    max_height = max(
        1,
        int(round(max(height_a, height_b)))
    )

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width,
            max_height
        )
    )

    return warped


# ============================================================
# CLEAN SEGMENTATION MASK
# ============================================================

def clean_mask(mask):

    """
    Clean segmentation noise and fill small holes.
    """

    close_kernel = np.ones(
        (7, 7),
        dtype=np.uint8
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    open_kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    return cleaned


# ============================================================
# FIND FOUR DOCUMENT CORNERS
# ============================================================

def find_document_corners(contour):

    """
    Try multiple polygon approximation strengths.

    Returns:
        4 corner points if successful
        None otherwise
    """

    perimeter = cv2.arcLength(
        contour,
        True
    )

    epsilon_factors = [
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.04,
        0.05
    ]

    for epsilon_factor in epsilon_factors:

        approx = cv2.approxPolyDP(
            contour,
            epsilon_factor * perimeter,
            True
        )

        if len(approx) == 4:

            points = approx.reshape(
                4,
                2
            ).astype(np.float32)

            return points

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Extract a document from an image using "
            "the trained DeepVerify segmentation model."
        )
    )

    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to the input document image"
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained segmentation model"
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where extraction results are saved"
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    output_dir = Path(args.output)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        "Using device:",
        device
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load U-Net model
    # --------------------------------------------------------

    print(
        "Loading segmentation model..."
    )

    model = UNet().to(
        device
    )

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Load input image
    # --------------------------------------------------------

    print(
        "Processing image:",
        image_path
    )

    pil_image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    original_rgb = np.array(
        pil_image
    )

    original_cv = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR
    )

    original_height, original_width = (
        original_cv.shape[:2]
    )

    print(
        "Original image size:",
        f"{original_width}x{original_height}"
    )

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),

            transforms.ToTensor()
        ]
    )

    input_tensor = (
        transform(
            pil_image
        )
        .unsqueeze(0)
        .to(device)
    )

    # --------------------------------------------------------
    # Predict segmentation mask
    # --------------------------------------------------------

    print(
        "Predicting document mask..."
    )

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probability = torch.sigmoid(
            logits
        )

        prediction = (
            probability
            > MASK_THRESHOLD
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

    # --------------------------------------------------------
    # Resize mask to original resolution
    # --------------------------------------------------------

    mask = cv2.resize(
        mask,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Clean mask
    # --------------------------------------------------------

    mask = clean_mask(
        mask
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        print(
            "ERROR: No document detected."
        )

        return

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

    print(
        f"Detected document area: "
        f"{area_ratio * 100:.2f}% "
        f"of image"
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if area_ratio < 0.01:

        print(
            "WARNING: Detected region is extremely small."
        )

    if area_ratio > 0.95:

        print(
            "WARNING: Segmentation selected almost "
            "the entire image."
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    visualization = (
        original_cv.copy()
    )

    # --------------------------------------------------------
    # Find document corners
    # --------------------------------------------------------

    points = find_document_corners(
        largest_contour
    )

    if points is not None:

        print(
            "4 document corners detected."
        )

        points_int = points.astype(
            np.int32
        )

        cv2.polylines(
            visualization,
            [points_int],
            True,
            (0, 255, 0),
            4
        )

        extracted = perspective_transform(
            original_cv,
            points
        )

    else:

        print(
            "Exact 4 corners not found."
        )

        print(
            "Using minimum-area rotated rectangle."
        )

        rotated_rect = cv2.minAreaRect(
            largest_contour
        )

        box = cv2.boxPoints(
            rotated_rect
        )

        box = box.astype(
            np.float32
        )

        box_int = box.astype(
            np.int32
        )

        cv2.polylines(
            visualization,
            [box_int],
            True,
            (0, 255, 0),
            4
        )

        extracted = perspective_transform(
            original_cv,
            box
        )

    # --------------------------------------------------------
    # Validate extracted image
    # --------------------------------------------------------

    if extracted is None or extracted.size == 0:

        print(
            "ERROR: Document extraction failed."
        )

        return

    # --------------------------------------------------------
    # Orientation correction
    # --------------------------------------------------------

    extracted_height, extracted_width = (
        extracted.shape[:2]
    )

    if extracted_height > extracted_width:

        extracted = cv2.rotate(
            extracted,
            cv2.ROTATE_90_CLOCKWISE
        )

        print(
            "Rotated extracted document "
            "to landscape orientation."
        )

    print(
        "Extracted document size:",
        f"{extracted.shape[1]}x"
        f"{extracted.shape[0]}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    mask_output = (
        output_dir
        / "predicted_mask.png"
    )

    detected_output = (
        output_dir
        / "detected_document.jpg"
    )

    extracted_output = (
        output_dir
        / "extracted_document.jpg"
    )

    cv2.imwrite(
        str(mask_output),
        mask
    )

    cv2.imwrite(
        str(detected_output),
        visualization
    )

    cv2.imwrite(
        str(extracted_output),
        extracted
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "DOCUMENT EXTRACTION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nSaved:"
    )

    print(
        mask_output
    )

    print(
        detected_output
    )

    print(
        extracted_output
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()