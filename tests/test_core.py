import numpy as np

from src.copy_move import detect_copy_move
from src.dataset import build_transforms
from src.extract_fields import clean_text
from src.preprocessing import error_level_analysis


def test_transforms_return_three_channel_tensor():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    transformed = build_transforms(224, augment=False)(image)
    assert tuple(transformed.shape) == (3, 224, 224)


def test_ocr_text_cleaning_normalizes_whitespace():
    assert clean_text("  Date   of\n birth  ") == "Date of birth"


def test_ela_returns_image_and_score(tmp_path):
    from PIL import Image

    path = tmp_path / "sample.jpg"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(path)
    import cv2

    image = cv2.imread(str(path))
    output = error_level_analysis(image)
    assert output.shape == image.shape
    assert output.dtype == image.dtype


def test_copy_move_handles_small_image(tmp_path):
    from PIL import Image

    path = tmp_path / "small.jpg"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
    score = detect_copy_move(str(path))
    assert isinstance(score, (int, float))
