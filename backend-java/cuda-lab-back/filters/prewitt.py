# filters/prewitt.py
# Prewitt Filter: Edge detector based on first derivative
# Complete implementation with separable CUDA

import numpy as np
from typing import Tuple, Dict

# Import shared CUDA initialization
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cuda_kernels import _initialize_cuda

# CUDA code for separable Prewitt
PREWITT_CUDA_SRC = r"""
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ size_t IDX(int x, int y, int w){
    return (size_t)y * w + x;
}

// Vertical sum box filter style
__global__ void box_vert_u8_to_f(const unsigned char* __restrict__ gray,
                                 float* __restrict__ V,
                                 int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int j = -r; j <= r; ++j){
        int yy = clampi(y + j, 0, h - 1);
        acc += gray[IDX(x,yy,w)];
    }

    V[IDX(x,y,w)] = acc;
}

// Calculate gx from V (vertical sum)
__global__ void prewitt_x_from_V(const float* __restrict__ V,
                                 float* __restrict__ gx,
                                 int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int i = -r; i <= r; ++i){
        int xx = clampi(x + i, 0, w - 1);
        int sx = (i < 0 ? -1 : (i > 0 ? +1 : 0));
        acc += V[IDX(xx,y,w)] * (float)sx;
    }

    gx[IDX(x,y,w)] = acc;
}

// Horizontal sum box filter style
__global__ void box_horiz_u8_to_f(const unsigned char* __restrict__ gray,
                                  float* __restrict__ H,
                                  int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int i = -r; i <= r; ++i){
        int xx = clampi(x + i, 0, w - 1);
        acc += gray[IDX(xx,y,w)];
    }

    H[IDX(x,y,w)] = acc;
}

// Calculate gy from H (horizontal sum)
__global__ void prewitt_y_from_H(const float* __restrict__ H,
                                 float* __restrict__ gy,
                                 int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int j = -r; j <= r; ++j){
        int yy = clampi(y + j, 0, h - 1);
        int sy = (j < 0 ? -1 : (j > 0 ? +1 : 0));
        acc += H[IDX(x,yy,w)] * (float)sy;
    }

    gy[IDX(x,y,w)] = acc;
}

// Final combination |gx| + |gy| and normalization
__global__ void combine_mag_to_gray(const float* __restrict__ gx,
                                    const float* __restrict__ gy,
                                    unsigned char* __restrict__ gray_out,
                                    int w, int h, int N, float gain)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float mag = fabsf(gx[IDX(x,y,w)]) + fabsf(gy[IDX(x,y,w)]);
    float v = (mag * gain) / (float)(N * (long long)N);

    if (v < 0.f) v = 0.f;
    else if (v > 255.f) v = 255.f;

    unsigned char u = (unsigned char)(v + 0.5f);
    gray_out[IDX(x,y,w)] = u;
}

} // extern C
"""

# Compiled module (lazy)
_prewitt_mod = None
_prewitt_compiled = False

def _ensure_prewitt_compiled():
    """Compile Prewitt kernels if not already compiled."""
    global _prewitt_mod, _prewitt_compiled
    if not _prewitt_compiled:
        # Initialize CUDA first (shared context)
        _initialize_cuda()
        from cuda_kernels import compile_cuda_kernel_to_ptx
        import pycuda.driver as drv
        # Compile using nvcc directly to avoid auto-detection issues
        ptx_code = compile_cuda_kernel_to_ptx(PREWITT_CUDA_SRC, arch="sm_89")
        _prewitt_mod = drv.module_from_buffer(ptx_code.encode())
        _prewitt_compiled = True

def apply_prewitt_cuda(
    image: np.ndarray,
    block_dim: Tuple[int, int],
    grid_dim: Tuple[int, int],
    gain: float = 8.0,
    mask_size: int = 3
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply Prewitt filter using separable CUDA approach.
    
    Args:
        image: Grayscale image (float32), shape (H, W)
        block_dim: (blockX, blockY)
        grid_dim: Ignored, calculated automatically
        gain: Edge enhancement factor (default 8.0)
        mask_size: Mask size NxN (default 3, must be odd)
    
    Returns:
        (result_image, timings_dict)
    """
    import pycuda.driver as cuda
    
    _ensure_prewitt_compiled()
    
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
    d_gray = cuda.mem_alloc(bytesGray)
    d_V = cuda.mem_alloc(Npix * 4)
    d_H = cuda.mem_alloc(Npix * 4)
    d_gx = cuda.mem_alloc(Npix * 4)
    d_gy = cuda.mem_alloc(Npix * 4)
    d_out = cuda.mem_alloc(bytesGray)
    
    # Copy to GPU
    cuda.memcpy_htod(d_gray, gray)
    
    # Get functions
    boxV = _prewitt_mod.get_function("box_vert_u8_to_f")
    gxF = _prewitt_mod.get_function("prewitt_x_from_V")
    boxH = _prewitt_mod.get_function("box_horiz_u8_to_f")
    gyF = _prewitt_mod.get_function("prewitt_y_from_H")
    comb = _prewitt_mod.get_function("combine_mag_to_gray")
    
    # Measure time
    start = cuda.Event()
    stop = cuda.Event()
    start.record()
    
    # Execute kernels
    boxV(d_gray, d_V, np.int32(w), np.int32(h), np.int32(N), block=block, grid=grid)
    gxF(d_V, d_gx, np.int32(w), np.int32(h), np.int32(N), block=block, grid=grid)
    boxH(d_gray, d_H, np.int32(w), np.int32(h), np.int32(N), block=block, grid=grid)
    gyF(d_H, d_gy, np.int32(w), np.int32(h), np.int32(N), block=block, grid=grid)
    comb(d_gx, d_gy, d_out, np.int32(w), np.int32(h), np.int32(N), np.float32(gain), block=block, grid=grid)
    
    stop.record()
    stop.synchronize()
    elapsed_ms = start.time_till(stop)
    
    # Copy result
    out = np.empty(bytesGray, dtype=np.uint8)
    cuda.memcpy_dtoh(out, d_out)
    
    result = out.reshape(h, w).astype(np.float32)
    
    timings = {
        "execution_time_ms": float(elapsed_ms),
        "kernel_time_ms": float(elapsed_ms),
    }
    
    return result, timings
