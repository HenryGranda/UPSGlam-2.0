# filters/gaussian.py
# Gaussian Filter: Smoothing with gaussian distribution
# Complete CUDA implementation with separable convolution

import numpy as np
from typing import Tuple, Dict

# Import shared CUDA initialization
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cuda_kernels import _initialize_cuda

# CUDA code for separable Gaussian filter
GAUSSIAN_CUDA_SRC = r"""
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

// Convert uint8 to float
__global__ void u8_to_f(const unsigned char* __restrict__ in_u8,
                        float* __restrict__ out_f,
                        int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    out_f[y * w + x] = (float)in_u8[y * w + x];
}

// Gaussian horizontal 1D convolution
__global__ void gauss_horiz_f(const float* __restrict__ in,
                              float* __restrict__ tmp,
                              int w, int h,
                              const float* __restrict__ k1d, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;
    int row_offset = y * w;

    for(int i = -r; i <= r; ++i){
        int xx = clampi(x + i, 0, w - 1);
        acc += in[row_offset + xx] * k1d[i + r];
    }
    tmp[row_offset + x] = acc;
}

// Gaussian vertical 1D convolution
__global__ void gauss_vert_f(const float* __restrict__ tmp,
                             float* __restrict__ out,
                             int w, int h,
                             const float* __restrict__ k1d, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for(int j = -r; j <= r; ++j){
        int yy = clampi(y + j, 0, h - 1);
        acc += tmp[yy * w + x] * k1d[j + r];
    }
    
    int out_idx = y * w + x;
    if (out_idx < w * h) {
        out[out_idx] = acc;
    }
}

// Convert float to uint8 with clamp
__global__ void f_to_u8(const float* __restrict__ in_f,
                        unsigned char* __restrict__ out_u8,
                        int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    if (idx >= w * h) return;
    
    float v = in_f[idx];
    v = fmaxf(0.0f, fminf(255.0f, v));
    out_u8[idx] = (unsigned char)(v + 0.5f);
}

} // extern C
"""

# Compiled module (lazy)
_gaussian_mod = None
_gaussian_compiled = False

def _ensure_gaussian_compiled():
    """Compile Gaussian kernels if not already compiled."""
    global _gaussian_mod, _gaussian_compiled
    if not _gaussian_compiled:
        # Initialize CUDA first (shared context)
        _initialize_cuda()
        from cuda_kernels import compile_cuda_kernel_to_ptx
        import pycuda.driver as drv
        # Compile using nvcc directly to avoid auto-detection issues
        ptx_code = compile_cuda_kernel_to_ptx(GAUSSIAN_CUDA_SRC, arch="sm_89")
        _gaussian_mod = drv.module_from_buffer(ptx_code.encode())
        _gaussian_compiled = True


def make_gauss_1d(N: int, sigma: float = None) -> np.ndarray:
    """
    Build a 1D Gaussian kernel normalized to sum to 1.
    
    Args:
        N: Kernel size (must be odd)
        sigma: Standard deviation. If None, calculated as N / 6.0
    
    Returns:
        1D float32 kernel of size N
    """
    if sigma is None:
        sigma = N / 6.0
    
    r = N // 2
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    k = np.empty(N, dtype=np.float32)
    s = 0.0
    for i in range(-r, r + 1):
        v = np.exp(-(i * i) * inv2s2)
        k[i + r] = v
        s += v
    k /= s
    return k.astype(np.float32)


def apply_gaussian_cuda(
    image: np.ndarray,
    block_dim: Tuple[int, int],
    grid_dim: Tuple[int, int],
    mask_size: int = 3,
    sigma: float = None,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply Gaussian filter using separable CUDA convolution.
    
    Args:
        image: Grayscale image (float32), shape (H, W)
        block_dim: (blockX, blockY)
        grid_dim: Ignored, calculated automatically
        mask_size: Kernel size N (default 3, must be odd)
        sigma: Standard deviation (default None = mask_size/6)
    
    Returns:
        (result_image, timings_dict)
    """
    import pycuda.driver as cuda
    
    _ensure_gaussian_compiled()
    
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
    
    # Build 1D Gaussian kernel
    k1d = make_gauss_1d(N, sigma)
    
    # Allocate GPU memory
    d_u8 = cuda.mem_alloc(bytesGray)
    d_in = cuda.mem_alloc(Npix * 4)
    d_tmp = cuda.mem_alloc(Npix * 4)
    d_out = cuda.mem_alloc(Npix * 4)
    d_k1d = cuda.mem_alloc(N * 4)
    d_result = cuda.mem_alloc(bytesGray)
    
    # Copy to GPU
    cuda.memcpy_htod(d_u8, gray)
    cuda.memcpy_htod(d_k1d, k1d)
    
    # Get functions
    u8_to_f = _gaussian_mod.get_function("u8_to_f")
    gauss_horiz = _gaussian_mod.get_function("gauss_horiz_f")
    gauss_vert = _gaussian_mod.get_function("gauss_vert_f")
    f_to_u8 = _gaussian_mod.get_function("f_to_u8")
    
    # Measure time
    start = cuda.Event()
    stop = cuda.Event()
    start.record()
    
    # Execute kernels: u8->float, horiz, vert, float->u8
    u8_to_f(d_u8, d_in, np.int32(w), np.int32(h), block=block, grid=grid)
    gauss_horiz(d_in, d_tmp, np.int32(w), np.int32(h), d_k1d, np.int32(N), block=block, grid=grid)
    gauss_vert(d_tmp, d_out, np.int32(w), np.int32(h), d_k1d, np.int32(N), block=block, grid=grid)
    f_to_u8(d_out, d_result, np.int32(w), np.int32(h), block=block, grid=grid)
    
    stop.record()
    stop.synchronize()
    elapsed_ms = start.time_till(stop)
    
    # Copy result
    out = np.empty(bytesGray, dtype=np.uint8)
    cuda.memcpy_dtoh(out, d_result)
    
    result = out.reshape(h, w).astype(np.float32)
    
    timings = {
        "execution_time_ms": float(elapsed_ms),
        "kernel_time_ms": float(elapsed_ms),
    }
    
    return result, timings


# Legacy function for backward compatibility with generic kernel approach
ALLOWED_SIZES = [3, 5, 7, 9, 21]

def gaussian_kernel(mask_size: int, sigma: float = None) -> np.ndarray:
    """
    Creates a 2D Gaussian kernel.
    
    Args:
        mask_size: Kernel size (must be odd)
        sigma: Standard deviation. If None, calculated as mask_size/6
    
    Returns:
        Normalized Gaussian kernel (sum = 1)
    """
    if mask_size not in ALLOWED_SIZES or mask_size % 2 == 0:
        raise ValueError(
            f"Invalid mask_size {mask_size} for gaussian. Allowed odd sizes: {ALLOWED_SIZES}"
        )

    if sigma is None:
        sigma = mask_size / 6.0
    
    # Kernel center
    center = mask_size // 2
    kernel = np.zeros((mask_size, mask_size), dtype=np.float32)
    
    # Generate gaussian values
    for i in range(mask_size):
        for j in range(mask_size):
            x = j - center
            y = i - center
            kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    
    # Normalize to sum to 1
    kernel = kernel / np.sum(kernel)
    return kernel
