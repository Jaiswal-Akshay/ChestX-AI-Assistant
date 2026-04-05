import numpy as np
from PIL import Image


def is_too_small(img: Image.Image, min_size: int = 100) -> bool:
    w, h = img.size
    return w < min_size or h < min_size


def is_blank_image(img: Image.Image, std_threshold: float = 5.0) -> bool:
    arr = np.array(img)
    return arr.std() < std_threshold


def is_low_contrast(img: Image.Image, contrast_threshold: float = 15.0) -> bool:
    arr = np.array(img)
    return (arr.max() - arr.min()) < contrast_threshold


def check_image_quality(
    img: Image.Image,
    min_size: int = 100,
    std_threshold: float = 5.0,
    contrast_threshold: float = 15.0,
) -> dict:
    """
    Return a dictionary of quality flags for one PIL image.
    """
    return {
        "too_small": is_too_small(img, min_size=min_size),
        "blank": is_blank_image(img, std_threshold=std_threshold),
        "low_contrast": is_low_contrast(img, contrast_threshold=contrast_threshold),
    }