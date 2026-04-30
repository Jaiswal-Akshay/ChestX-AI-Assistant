from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image


def is_too_small(img: Image.Image, min_size: int = 100) -> bool:
    """Return True if either image dimension is below min_size."""
    width, height = img.size
    return width < min_size or height < min_size


def is_blank_image(img: Image.Image, std_threshold: float = 5.0) -> bool:
    """Return True if grayscale pixel standard deviation is too low."""
    arr = np.asarray(img, dtype=np.float32)
    return float(arr.std()) < std_threshold


def is_low_contrast(img: Image.Image, contrast_threshold: float = 15.0) -> bool:
    """Return True if the image intensity range is too narrow."""
    arr = np.asarray(img, dtype=np.float32)
    return float(arr.max() - arr.min()) < contrast_threshold


def check_image_quality(
    img: Image.Image,
    min_size: int = 100,
    std_threshold: float = 5.0,
    contrast_threshold: float = 15.0,
) -> Dict[str, bool]:
    """
    Return a dictionary containing image quality flags.
    """
    return {
        "too_small": is_too_small(img, min_size=min_size),
        "blank": is_blank_image(img, std_threshold=std_threshold),
        "low_contrast": is_low_contrast(img, contrast_threshold=contrast_threshold),
    }