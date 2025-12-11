import cv2
import numpy as np


def apply_laplacian(image: np.ndarray) -> np.ndarray:
    """
    Simple Laplacian edge detector on the CPU.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
