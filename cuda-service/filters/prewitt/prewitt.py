import cv2
import numpy as np


def apply_prewitt(image: np.ndarray) -> np.ndarray:
    """
    Classic Prewitt operator combined magnitude result.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]], dtype=np.float32)
    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, kernelx)
    gy = cv2.filter2D(gray, -1, kernely)
    mag = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
