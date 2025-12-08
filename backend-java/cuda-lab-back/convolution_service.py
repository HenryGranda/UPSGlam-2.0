from filters import get_filter_kernel
from image_utils import decode_image_base64, encode_image_base64, decode_image_bytes, encode_image_bytes
from cuda_kernels import convolve_gpu_single
import numpy as np


# Default configurations for each filter
FILTER_CONFIGS = {
    "prewitt": {"mask_size": 3, "gain": 8.0},
    "laplacian": {"mask_size": 3},
    "gaussian": {"mask_size": 121},
    "box_blur": {"mask_size": 91},
    "ups_logo": {"mask_size": 5},
    "ups_color": {"mask_size": 1}
}


def process_convolution_request(payload: dict) -> dict:
    """
    Main orchestrator - delegates to each filter implementation.
    
    LEGACY: For cuda-lab-back compatibility. Uses base64 encoding and grayscale.
    """
    image_b64 = payload["image_base64"]
    filter_conf = payload["filter"]
    cuda_conf = payload["cuda_config"]

    filter_type = filter_conf["type"]
    mask_size = int(filter_conf["mask_size"])
    gain = float(filter_conf.get("gain", 8.0))  # Para Prewitt

    block_dim = tuple(cuda_conf["block_dim"])
    grid_dim = tuple(cuda_conf["grid_dim"])

    img_np = decode_image_base64(image_b64)
    height, width = img_np.shape

    # Get filter
    filter_info = get_filter_kernel(filter_type, mask_size)
    filter_used = filter_info["type"]
    mask_size_used = filter_info["mask_size_used"]

    # Execute filter based on type
    if filter_used == "prewitt":
        # Prewitt has its own complete CUDA function in filters/prewitt.py
        prewitt_func = filter_info["cuda_function"]
        result_np, timings = prewitt_func(img_np, block_dim, grid_dim, gain=gain, mask_size=mask_size_used)
    elif filter_used == "laplacian":
        # Laplacian has its own complete CUDA function in filters/laplacian.py
        laplacian_func = filter_info["cuda_function"]
        result_np, timings = laplacian_func(img_np, block_dim, grid_dim, mask_size=mask_size_used)
    elif filter_used == "gaussian":
        # Gaussian has its own complete CUDA function in filters/gaussian.py
        gaussian_func = filter_info["cuda_function"]
        result_np, timings = gaussian_func(img_np, block_dim, grid_dim, mask_size=mask_size_used)
    elif filter_used == "box_blur":
        # Box Blur has its own complete CUDA function in filters/box_blur.py
        box_blur_func = filter_info["cuda_function"]
        result_np, timings = box_blur_func(img_np, block_dim, grid_dim, mask_size=mask_size_used)
    else:
        # Fallback to generic kernel (should not happen)
        result_np, timings = convolve_gpu_single(
            img_np,
            filter_info["kernel"],
            block_dim,
            grid_dim,
        )

    # Encode result
    result_b64 = encode_image_base64(result_np)

    return {
        "status": "ok",
        "result_image_base64": result_b64,
        "execution_time_ms": float(timings.get("execution_time_ms", 0.0)),
        "kernel_time_ms": float(timings.get("kernel_time_ms", 0.0)),
        "image_width": width,
        "image_height": height,
        "filter_used": filter_used,
        "mask_size_used": mask_size_used,
        "block_dim": list(block_dim),
        "grid_dim": list(grid_dim),
    }


def process_convolution_bytes(
    image_bytes: bytes,
    filter_name: str,
    preserve_color: bool = True
) -> bytes:
    """
    Process image bytes with filter and return filtered image bytes.
    
    UPSGlam function: Receives raw bytes, returns raw bytes (JPEG/PNG).
    Supports RGB images (applies filter to each channel separately).
    
    Args:
        image_bytes: Raw image bytes (JPEG/PNG)
        filter_name: Filter to apply (prewitt, laplacian, gaussian, box_blur, ups_logo, ups_color)
        preserve_color: If True, process RGB channels separately. If False, convert to grayscale
    
    Returns:
        bytes: Filtered image as JPEG bytes
    """
    # Get default config for filter
    config = FILTER_CONFIGS.get(filter_name, {"mask_size": 3})
    mask_size = config["mask_size"]
    gain = config.get("gain", 8.0)
    
    # Decode image
    img_np = decode_image_bytes(image_bytes, preserve_color=preserve_color)
    
    # Auto-calculate CUDA dimensions
    if img_np.ndim == 3:
        height, width, channels = img_np.shape
    else:
        height, width = img_np.shape
        channels = 1
    
    block_dim = (16, 16)
    grid_dim = (
        (width + block_dim[0] - 1) // block_dim[0],
        (height + block_dim[1] - 1) // block_dim[1]
    )
    
    # Get filter
    filter_info = get_filter_kernel(filter_name, mask_size)
    filter_used = filter_info["type"]
    mask_size_used = filter_info["mask_size_used"]
    
    # Process RGB image channel by channel
    if img_np.ndim == 3:
        result_channels = []
        
        for c in range(3):  # R, G, B
            channel = img_np[:, :, c]
            
            # Apply filter to channel
            if filter_used == "prewitt":
                prewitt_func = filter_info["cuda_function"]
                result_channel, _ = prewitt_func(channel, block_dim, grid_dim, gain=gain, mask_size=mask_size_used)
            elif filter_used == "laplacian":
                laplacian_func = filter_info["cuda_function"]
                result_channel, _ = laplacian_func(channel, block_dim, grid_dim, mask_size=mask_size_used)
            elif filter_used == "gaussian":
                gaussian_func = filter_info["cuda_function"]
                result_channel, _ = gaussian_func(channel, block_dim, grid_dim, mask_size=mask_size_used)
            elif filter_used == "box_blur":
                box_blur_func = filter_info["cuda_function"]
                result_channel, _ = box_blur_func(channel, block_dim, grid_dim, mask_size=mask_size_used)
            else:
                # Fallback
                result_channel, _ = convolve_gpu_single(channel, filter_info["kernel"], block_dim, grid_dim)
            
            result_channels.append(result_channel)
        
        # Stack channels back together
        result_np = np.stack(result_channels, axis=2)
    
    else:
        # Grayscale processing
        if filter_used == "prewitt":
            prewitt_func = filter_info["cuda_function"]
            result_np, _ = prewitt_func(img_np, block_dim, grid_dim, gain=gain, mask_size=mask_size_used)
        elif filter_used == "laplacian":
            laplacian_func = filter_info["cuda_function"]
            result_np, _ = laplacian_func(img_np, block_dim, grid_dim, mask_size=mask_size_used)
        elif filter_used == "gaussian":
            gaussian_func = filter_info["cuda_function"]
            result_np, _ = gaussian_func(img_np, block_dim, grid_dim, mask_size=mask_size_used)
        elif filter_used == "box_blur":
            box_blur_func = filter_info["cuda_function"]
            result_np, _ = box_blur_func(img_np, block_dim, grid_dim, mask_size=mask_size_used)
        else:
            result_np, _ = convolve_gpu_single(img_np, filter_info["kernel"], block_dim, grid_dim)
    
    # Encode result to JPEG bytes
    result_bytes = encode_image_bytes(result_np, format='JPEG', quality=95)
    
    return result_bytes
