import numpy as np
from PIL import Image, ImageChops


def ela_analysis(image_path, quality=90):
    original = Image.open(image_path).convert("RGB")

    temp_path = "temp.jpg"
    original.save(temp_path, "JPEG", quality=quality)

    compressed = Image.open(temp_path)

    from PIL import ImageEnhance

    difference = ImageChops.difference(original, compressed)

    enhancer = ImageEnhance.Brightness(difference)

    difference = enhancer.enhance(25)

    difference.save("ela_result.jpg")

    diff_array = np.array(difference)

    score = float(np.mean(diff_array))

    return score, difference