# filters/ups_color.py
"""
UPS Color Filter - Creative Filter #2 (Required for Project)

Applies color tinting with UPS corporate colors (gold and maroon).
This filter converts image to sepia-tone with UPS branding colors.
"""

import numpy as np


def ups_color_cuda(img_np: np.ndarray, block_dim: tuple, grid_dim: tuple, mask_size: int = 3) -> tuple:
    """
    Apply UPS Color filter: Sepia tone with UPS corporate gold tint.
    
    Strategy:
    1. Convert to intensity-based representation
    2. Apply warm golden tint (UPS gold: #C69214)
    3. Add slight contrast enhancement
    4. Preserve highlights and shadows
    
    UPS Corporate Colors:
    - Primary Gold: #C69214 (R:198, G:146, B:20) → Grayscale equivalent: ~150
    - Maroon: #862633 → Grayscale equivalent: ~100
    
    For grayscale images, we create a "warm" effect by:
    - Boosting mid-tones (golden look)
    - Darkening shadows slightly (maroon depth)
    - Preserving highlights
    
    Args:
        img_np: Input grayscale image (H, W) as float32
        block_dim: CUDA block dimensions (not used for CPU-based color mapping)
        grid_dim: CUDA grid dimensions (not used for CPU-based color mapping)
        mask_size: Unused, kept for API compatibility
    
    Returns:
        (result_np, timings): Filtered image and performance metrics
    """
    import time
    start_time = time.perf_counter()
    
    # Create result array
    result_np = img_np.copy()
    
    # Step 1: Normalize to [0, 1]
    normalized = result_np / 255.0
    
    # Step 2: Apply UPS golden tone curve
    # This creates a warm, vintage look with golden mid-tones
    
    # Shadows (0.0 - 0.3): Slightly darken, add maroon depth
    shadow_mask = normalized < 0.3
    result_np[shadow_mask] = normalized[shadow_mask] * 0.85 * 255.0
    
    # Mid-tones (0.3 - 0.7): Boost with golden glow
    # Golden color in grayscale: emphasize values around 150-180
    midtone_mask = (normalized >= 0.3) & (normalized < 0.7)
    midtone_values = normalized[midtone_mask]
    # Apply S-curve for golden warmth
    golden_curve = np.power(midtone_values, 0.85) * 1.15
    result_np[midtone_mask] = np.clip(golden_curve * 255.0, 0, 255)
    
    # Highlights (0.7 - 1.0): Preserve with slight warm tint
    highlight_mask = normalized >= 0.7
    highlight_values = normalized[highlight_mask]
    result_np[highlight_mask] = np.clip(highlight_values * 1.05 * 255.0, 0, 255)
    
    # Step 3: Add subtle contrast boost
    # Enhance contrast by stretching histogram slightly
    mean_val = np.mean(result_np)
    contrast_factor = 1.1
    result_np = (result_np - mean_val) * contrast_factor + mean_val
    
    # Step 4: Apply vignette effect (darker edges, like vintage photo)
    height, width = result_np.shape
    center_y, center_x = height // 2, width // 2
    
    # Create distance map from center
    y_coords, x_coords = np.ogrid[:height, :width]
    distance_from_center = np.sqrt(
        ((y_coords - center_y) / (height / 2)) ** 2 + 
        ((x_coords - center_x) / (width / 2)) ** 2
    )
    
    # Vignette mask: 1.0 at center, ~0.7 at edges
    vignette_mask = 1.0 - (distance_from_center * 0.3)
    vignette_mask = np.clip(vignette_mask, 0.7, 1.0)
    
    # Apply vignette
    result_np = result_np * vignette_mask
    
    # Final clipping
    result_np = np.clip(result_np, 0, 255).astype(np.float32)
    
    # Timing
    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) * 1000.0
    
    timings = {
        "execution_time_ms": execution_time_ms,
        "kernel_time_ms": 0.0,  # CPU-based, no CUDA kernel
        "filter_name": "ups_color"
    }
    
    return result_np, timings
