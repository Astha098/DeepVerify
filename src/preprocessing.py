"""
preprocessing.py
-----------------
Classical computer-vision preprocessing steps applied before the document
image is handed to the deep-learning models. This is deliberately built with
OpenCV only (no DL) because a real ID pipeline always needs a robust,
fast, non-learned first pass before the expensive models run.

Functions:
    order_points          - order 4 corner points [tl, tr, br, bl]
    find_document_contour - locate the largest 4-point contour (the document)
    perspective_correct   - warp a photographed document to a top-down view
    denoise               - remove sensor / compression noise
    normalize_brightness  - CLAHE-based brightness/contrast normalization
    preprocess_pipeline   - run the full preprocessing chain
    error_level_analysis  - classic ELA, a cheap tamper/edit indicator
"""

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_document_contour(image: np.ndarray):
    """
    Try to find the 4-point contour of a document in a photographed scene.
    Returns None if no confident quadrilateral is found (e.g. the image is
    already a tight crop of the document, which is common in ID datasets).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    img_area = image.shape[0] * image.shape[1]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * img_area:
            return approx.reshape(4, 2)
    return None


def perspective_correct(image: np.ndarray, target_size=(640, 400)) -> np.ndarray:
    """
    Detect the document quadrilateral and warp it to a flat, top-down view.
    Falls back to a plain resize if no reliable quad is found.
    """
    quad = find_document_contour(image)
    if quad is None:
        return cv2.resize(image, target_size)

    rect = order_points(quad.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < 10 or maxHeight < 10:
        return cv2.resize(image, target_size)

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return cv2.resize(warped, target_size)


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove sensor/JPEG noise while preserving text edges."""
    return cv2.fastNlMeansDenoisingColored(image, None, h=7, hColor=7,
                                            templateWindowSize=7, searchWindowSize=21)


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB space to correct uneven lighting/glare."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    lab = cv2.merge((lightness, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_pipeline(image: np.ndarray, target_size=(640, 400)) -> np.ndarray:
    """Full preprocessing chain: perspective correct -> denoise -> normalize."""
    img = perspective_correct(image, target_size)
    img = denoise(img)
    img = normalize_brightness(img)
    return img


def error_level_analysis(image_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    """
    Classic Error Level Analysis: re-compress the image at a known JPEG
    quality and diff against the original. Regions that were pasted/edited
    after the original compression tend to show a different error level
    than the untouched background. This is a lightweight, non-learned
    tamper *hint*, not a verdict -- it feeds into the anomaly score
    alongside the CNN-based detector.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image_bgr, encode_param)
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(image_bgr, recompressed)
    diff = cv2.convertScaleAbs(diff, alpha=10)  # amplify for visibility
    return diff
