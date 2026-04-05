import numpy as np
from PIL import Image

def is_blank_image(img, threshold=5):
    arr = np.array(img)
    return arr.std() < threshold

def is_low_contrast(img, threshold=15):
    arr = np.array(img)
    return (arr.max() - arr.min()) < threshold

def is_too_small(img, min_size=100):
    w, h = img.size
    return w < min_size or h < min_size