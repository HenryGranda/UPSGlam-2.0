# filters/__init__.py
# Central router for all convolution filters

from .box_blur import box_blur_kernel, apply_box_blur_cuda
from .gaussian import gaussian_kernel, apply_gaussian_cuda
from .laplacian import laplacian_kernel, apply_laplacian_cuda
from .prewitt import apply_prewitt_cuda
from .ups_logo import apply_ups_logo_bytes
from .ups_color import ups_color_cuda
from .boomerang import apply_boomerang_bytes


def get_filter_kernel(filter_type: str, mask_size: int) -> dict:
    """
    Returns the necessary information to apply a convolution filter.
    
    This function is the main entry point from convolution_service.py.
    Selects the appropriate filter and returns the necessary kernels.
    
    Args:
        filter_type: Filter type ("box_blur", "gaussian", "laplacian", "prewitt", "ups_logo", "ups_color")
        mask_size: Desired kernel size (must be odd for most filters)
    
    Returns:
        dict with the following keys:
            - "type": normalized filter name
            - "kernel": numpy array for single-kernel filters (box_blur, gaussian, laplacian)
            - "kernel_x", "kernel_y": numpy arrays for Prewitt (two kernels)
            - "mask_size_used": actual size used (may differ from requested)
    
    Raises:
        ValueError: If filter_type is not recognized or mask_size is invalid
    
    Implemented filters:
        - box_blur: Simple average (smoothing)
        - gaussian: Gaussian smoothing (better quality than box_blur)
        - laplacian: Edge detection (second derivative)
        - prewitt: Directional edge detection (first derivative, Gx and Gy)
        - ups_logo: Creative filter with UPS logo overlay (Project requirement)
        - ups_color: Creative filter with UPS color tinting (Project requirement)
    """
    ft = filter_type.lower()

    if ft == "box_blur":
        # Box Blur uses separable CUDA implementation
        if mask_size % 2 == 0:
            raise ValueError(f"Box Blur mask_size must be odd, got {mask_size}")
        
        return {
            "type": "box_blur",
            "cuda_function": apply_box_blur_cuda,
            "mask_size_used": mask_size,
        }

    if ft == "gaussian":
        # Gaussian uses separable CUDA implementation
        if mask_size % 2 == 0:
            raise ValueError(f"Gaussian mask_size must be odd, got {mask_size}")
        
        return {
            "type": "gaussian",
            "cuda_function": apply_gaussian_cuda,
            "mask_size_used": mask_size,
        }

    if ft == "laplacian":
        # Laplacian supports both 3x3 (classic) and NxN (LoG)
        if mask_size % 2 == 0:
            raise ValueError(f"Laplacian mask_size must be odd, got {mask_size}")
        
        return {
            "type": "laplacian",
            "cuda_function": apply_laplacian_cuda,
            "mask_size_used": mask_size,
        }

    if ft == "prewitt":
        # Prewitt uses complete separable CUDA implementation
        # Validate that mask_size is odd
        if mask_size % 2 == 0:
            raise ValueError(f"Prewitt mask_size must be odd, got {mask_size}")
        
        return {
            "type": "prewitt",
            "cuda_function": apply_prewitt_cuda,
            "mask_size_used": mask_size,
        }
    
    if ft == "ups_logo":
        # UPS Logo: Logo overlay with aura effects (Creative filter #1)
        # Note: This filter is handled directly in convolution_service.py as it works with bytes
        return {
            "type": "ups_logo",
            "cuda_function": apply_ups_logo_bytes,
            "mask_size_used": 5,  # N/A for this filter
        }
    
    if ft == "ups_color":
        # UPS Color: Sepia tone with UPS corporate colors (Creative filter #2)
        return {
            "type": "ups_color",
            "cuda_function": ups_color_cuda,
            "mask_size_used": 1,  # Color mapping, no convolution
        }

    raise ValueError(f"Unknown filter type: {filter_type}")
