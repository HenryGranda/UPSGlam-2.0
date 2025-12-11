import cv2
import numpy as np


def apply_gaussian(image: np.ndarray, ksize: int = 9, sigma: float = 2.0) -> np.ndarray:
    """
    Apply a gaussian blur; kernel size must be odd.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)
