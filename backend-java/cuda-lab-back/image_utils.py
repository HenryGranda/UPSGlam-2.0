# cuda-lab-back/image_utils.py

import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image


# ==================== BASE64 Functions (Legacy) ====================

def _strip_data_url_prefix(image_base64: str) -> str:
    """
    If the input is something like 'data:image/png;base64,AAAA...', remove the 'data:...base64,' prefix.
    """
    if "," in image_base64:
        return image_base64.split(",", 1)[1]
    return image_base64


def decode_image_base64(image_base64: str) -> np.ndarray:
    """
    Receives a base64 image string (possibly with data URL prefix)
    and returns a NumPy array in grayscale (float32) of shape (H, W).
    
    LEGACY: For cuda-lab-back compatibility. Use decode_image_bytes for UPSGlam.
    """
    if not image_base64:
        raise ValueError("image_base64 is empty")

    b64_data = _strip_data_url_prefix(image_base64)

    try:
        img_bytes = base64.b64decode(b64_data)
    except Exception:
        raise ValueError("Invalid base64 image string")

    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # L = grayscale
    img_np = np.array(img).astype(np.float32)
    return img_np


def encode_image_base64(img_np: np.ndarray) -> str:
    """
    Receives a np.ndarray (H, W), normalizes/clips it to [0, 255] uint8,
    converts it to PNG in memory, and returns 'data:image/png;base64,...'.
    
    LEGACY: For cuda-lab-back compatibility. Use encode_image_bytes for UPSGlam.
    """
    if img_np.ndim != 2:
        raise ValueError("Expected 2D array for grayscale image")

    # Clip and convert to uint8
    img_clipped = np.clip(img_np, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_clipped, mode="L")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    b64_bytes = base64.b64encode(buffer.read())
    b64_str = b64_bytes.decode("utf-8")

    return f"data:image/png;base64,{b64_str}"


# ==================== BYTES Functions (UPSGlam) ====================

def decode_image_bytes(image_bytes: bytes, preserve_color: bool = True) -> np.ndarray:
    """
    Receives raw image bytes (JPEG/PNG) and returns NumPy array.
    
    Args:
        image_bytes: Raw image bytes from HTTP request body
        preserve_color: If True, returns RGB (H,W,3). If False, grayscale (H,W)
    
    Returns:
        np.ndarray: float32 array, shape (H,W,3) for RGB or (H,W) for grayscale
    
    UPSGlam: Use this for /filters/{filterName} endpoint
    """
    if not image_bytes:
        raise ValueError("image_bytes is empty")
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Invalid image bytes: {e}")
    
    # Convert to RGB or grayscale
    if preserve_color:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_np = np.array(img).astype(np.float32)  # shape: (H, W, 3)
    else:
        img = img.convert('L')  # grayscale
        img_np = np.array(img).astype(np.float32)  # shape: (H, W)
    
    return img_np


def encode_image_bytes(img_np: np.ndarray, format: str = 'JPEG', quality: int = 95) -> bytes:
    """
    Receives np.ndarray and returns raw image bytes (JPEG/PNG).
    
    Args:
        img_np: NumPy array, shape (H,W,3) for RGB or (H,W) for grayscale
        format: 'JPEG' or 'PNG'
        quality: JPEG quality (1-100), ignored for PNG
    
    Returns:
        bytes: Raw image bytes ready for HTTP response or Supabase upload
    
    UPSGlam: Use this for /filters/{filterName} endpoint
    """
    # Validate dimensions
    if img_np.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got {img_np.ndim}D")
    
    if img_np.ndim == 3 and img_np.shape[2] != 3:
        raise ValueError(f"Expected 3 channels for RGB, got {img_np.shape[2]}")
    
    # Clip and convert to uint8
    img_clipped = np.clip(img_np, 0, 255).astype(np.uint8)
    
    # Create PIL Image
    if img_np.ndim == 3:
        img = Image.fromarray(img_clipped, mode='RGB')
    else:
        img = Image.fromarray(img_clipped, mode='L')
    
    # Encode to bytes
    buffer = io.BytesIO()
    if format.upper() == 'JPEG':
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
    elif format.upper() == 'PNG':
        img.save(buffer, format='PNG', optimize=True)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    buffer.seek(0)
    return buffer.read()
