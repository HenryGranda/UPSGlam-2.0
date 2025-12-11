import cv2
import numpy as np


def apply_blox(image: np.ndarray, block_size: int = 12) -> np.ndarray:
    """
    Pixelates the image by averaging blocks of pixels.
    """
    h, w, _ = image.shape
    out = image.copy()
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            roi = image[y:y+block_size, x:x+block_size]
            avg_color = roi.mean(axis=(0, 1), keepdims=True)
            out[y:y+block_size, x:x+block_size] = avg_color
    return out
