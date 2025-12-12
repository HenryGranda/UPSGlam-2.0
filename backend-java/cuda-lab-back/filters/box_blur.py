# filters/box_blur.py
# Box Blur Filter: Simple neighborhood average
# Complete CUDA implementation with separable convolution

import numpy as np
from typing import Tuple, Dict

# Import shared CUDA initialization
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cuda_kernels import _initialize_cuda

# CUDA code for separable Box Blur
BOX_BLUR_CUDA_SRC = r"""
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

// Horizontal pass: uint8 -> float accumulation along X
__global__ void box_horiz_u8_to_f(const unsigned char* __restrict__ src,
                                  float* __restrict__ tmp,
                                  int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int i = -r; i <= r; ++i){
        int xx = clampi(x + i, 0, w - 1);
        acc += (float)src[y * w + xx];
    }

    tmp[y * w + x] = acc;
}

// Vertical pass: float -> uint8 normalized by N*N
__global__ void box_vert_f_to_u8(const float* __restrict__ tmp,
                                 unsigned char* __restrict__ dst,
                                 int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float invN = 1.0f / (float)N;
    float acc = 0.f;

    for (int j = -r; j <= r; ++j){
        int yy = clampi(y + j, 0, h - 1);
        acc += tmp[yy * w + x];
    }

    // Normalize by N*N (horizontal * vertical)
    int val = (int)(acc * invN * invN + 0.5f);
    val = val < 0 ? 0 : (val > 255 ? 255 : val);

    dst[y * w + x] = (unsigned char)val;
}

} // extern C
"""

# Compiled module (lazy)
_box_blur_mod = None
_box_blur_compiled = False

def _ensure_box_blur_compiled():
    """Compile Box Blur kernels if not already compiled."""
    global _box_blur_mod, _box_blur_compiled
    if not _box_blur_compiled:
        # Initialize CUDA first (shared context)
        _initialize_cuda()
        from cuda_kernels import compile_cuda_kernel_to_ptx
        import pycuda.driver as drv
        # Compile using nvcc directly to avoid auto-detection issues
        ptx_code = compile_cuda_kernel_to_ptx(BOX_BLUR_CUDA_SRC, arch="sm_89")
        _box_blur_mod = drv.module_from_buffer(ptx_code.encode())
        _box_blur_compiled = True


def apply_box_blur_cuda(
    image: np.ndarray,
    block_dim: Tuple[int, int],
    grid_dim: Tuple[int, int],
    mask_size: int = 3,
    passes: int = 1,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply Box Blur filter using separable CUDA convolution.
    
    Args:
        image: Grayscale image (float32), shape (H, W)
        block_dim: (blockX, blockY)
        grid_dim: Ignored, calculated automatically
        mask_size: Kernel size N (default 3, must be odd)
        passes: Number of blur passes (default 1)
    
    Returns:
        (result_image, timings_dict)
    """
    import pycuda.driver as cuda
    
    _ensure_box_blur_compiled()
    
    if image.ndim != 2:
        raise ValueError("Image must be 2D (grayscale)")
    
    if mask_size % 2 == 0:
        raise ValueError(f"mask_size must be odd, got {mask_size}")
    
    h, w = image.shape
    N = mask_size
    
    # Convert to uint8 if float32
    if image.dtype == np.float32:
        img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)
    
    gray = np.ascontiguousarray(img_u8.reshape(-1))
    Npix = w * h
    bytesGray = Npix
    
    # Calculate correct grid
    blockX, blockY = block_dim
    gridX = (w + blockX - 1) // blockX
    gridY = (h + blockY - 1) // blockY
    
    block = (blockX, blockY, 1)
    grid = (gridX, gridY, 1)
    
    # Allocate GPU memory
    d_in = cuda.mem_alloc(bytesGray)
    d_out = cuda.mem_alloc(bytesGray)
    d_tmp = cuda.mem_alloc(Npix * 4)  # float buffer
    
    # Copy to GPU
    cuda.memcpy_htod(d_in, gray)
    
    # Get functions
    box_horiz = _box_blur_mod.get_function("box_horiz_u8_to_f")
    box_vert = _box_blur_mod.get_function("box_vert_f_to_u8")
    
    # Measure time
    start = cuda.Event()
    stop = cuda.Event()
    start.record()
    
    # Execute multiple passes if requested
    for _ in range(passes):
        # Horizontal: uint8 -> float
        box_horiz(d_in, d_tmp, np.int32(w), np.int32(h), np.int32(N), block=block, grid=grid)
        
        # Vertical: float -> uint8
        box_vert(d_tmp, d_out, np.int32(w), np.int32(h), np.int32(N), block=block, grid=grid)
        
        # Swap buffers for next pass
        d_in, d_out = d_out, d_in
    
    stop.record()
    stop.synchronize()
    elapsed_ms = start.time_till(stop)
    
    # Copy result (it's in d_in after swap)
    out = np.empty(bytesGray, dtype=np.uint8)
    cuda.memcpy_dtoh(out, d_in)
    
    result = out.reshape(h, w).astype(np.float32)
    
    timings = {
        "execution_time_ms": float(elapsed_ms),
        "kernel_time_ms": float(elapsed_ms),
    }
    
    return result, timings


# Legacy function for backward compatibility with generic kernel approach
def box_blur_kernel(mask_size: int) -> np.ndarray:
    """
    Creates a Box Blur kernel (simple average).
    All values are 1/(mask_size^2).
    """
    kernel = np.ones((mask_size, mask_size), dtype=np.float32)
    kernel /= (mask_size * mask_size)
    return kernel
