"""
Progressive convolution service - processes image in chunks and yields intermediate results
for real-time visualization of pixel-by-pixel processing.
"""
import numpy as np
from filters import get_filter_kernel
from image_utils import encode_image_base64
import time
from typing import Generator, Tuple


def process_progressive_convolution(
    img_np: np.ndarray,
    filter_type: str,
    mask_size: int,
    gain: float,
    block_dim: Tuple[int, int],
    grid_dim: Tuple[int, int],
    chunk_size: int = 32  # Process in chunks of rows
) -> Generator[dict, None, None]:
    """
    Process convolution progressively, yielding intermediate results.
    
    Args:
        img_np: Input grayscale image
        filter_type: Type of filter to apply
        mask_size: Size of convolution mask
        gain: Gain parameter (for Prewitt)
        block_dim: CUDA block dimensions
        grid_dim: CUDA grid dimensions
        chunk_size: Number of rows to process per chunk
    
    Yields:
        dict with progress info and partial result image
    """
    height, width = img_np.shape
    filter_info = get_filter_kernel(filter_type, mask_size)
    filter_used = filter_info["type"]
    mask_size_used = filter_info["mask_size_used"]
    
    # Initialize result image (start with gray/black)
    result_np = np.zeros_like(img_np, dtype=np.float32)
    
    # Calculate total chunks
    total_chunks = (height + chunk_size - 1) // chunk_size
    
    start_time = time.time()
    
    # Process image in horizontal chunks (row by row simulation)
    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, height)
        
        # Extract chunk to process
        chunk_img = img_np[chunk_start:chunk_end, :]
        
        # Process this chunk with the full filter
        if filter_used == "prewitt":
            prewitt_func = filter_info["cuda_function"]
            chunk_result, _ = prewitt_func(chunk_img, block_dim, grid_dim, gain=gain, mask_size=mask_size_used)
        elif filter_used == "laplacian":
            laplacian_func = filter_info["cuda_function"]
            chunk_result, _ = laplacian_func(chunk_img, block_dim, grid_dim, mask_size=mask_size_used)
        elif filter_used == "gaussian":
            gaussian_func = filter_info["cuda_function"]
            chunk_result, _ = gaussian_func(chunk_img, block_dim, grid_dim, mask_size=mask_size_used)
        elif filter_used == "box_blur":
            box_blur_func = filter_info["cuda_function"]
            chunk_result, _ = box_blur_func(chunk_img, block_dim, grid_dim, mask_size=mask_size_used)
        else:
            # Fallback
            from cuda_kernels import convolve_gpu_single
            chunk_result, _ = convolve_gpu_single(chunk_img, filter_info["kernel"], block_dim, grid_dim)
        
        # Update result with processed chunk
        result_np[chunk_start:chunk_end, :] = chunk_result
        
        # Calculate progress
        progress = ((chunk_idx + 1) / total_chunks) * 100
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Encode current state of the image
        result_b64 = encode_image_base64(result_np)
        
        # Yield progress update
        yield {
            "progress": progress,
            "chunk": chunk_idx + 1,
            "total_chunks": total_chunks,
            "rows_processed": chunk_end,
            "total_rows": height,
            "elapsed_ms": elapsed_ms,
            "result_image_base64": result_b64,
            "filter_used": filter_used,
            "mask_size_used": mask_size_used,
        }
        
        # Small delay to make visualization visible (adjustable)
        time.sleep(0.05)  # 50ms delay between chunks for smooth animation
    
    # Final yield with completion status
    total_time = (time.time() - start_time) * 1000
    yield {
        "progress": 100,
        "chunk": total_chunks,
        "total_chunks": total_chunks,
        "rows_processed": height,
        "total_rows": height,
        "elapsed_ms": total_time,
        "result_image_base64": encode_image_base64(result_np),
        "filter_used": filter_used,
        "mask_size_used": mask_size_used,
        "completed": True,
    }
