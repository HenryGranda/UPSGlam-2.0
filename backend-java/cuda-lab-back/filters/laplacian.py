# filters/laplacian.py
# Laplacian Filter: Edge detector based on second derivative
# Complete CUDA implementation supporting variable kernel sizes

import numpy as np
from typing import Tuple, Dict

# Import shared CUDA initialization
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cuda_kernels import _initialize_cuda

# CUDA code for Laplacian and Laplacian of Gaussian (LoG)
LAPLACIAN_CUDA_SRC = r"""
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ size_t IDX(int x, int y, int w){
    return (size_t)y * w + x;
}

// Classic 3x3 Laplacian (8-neighbor)
// Kernel:
//   [-1 -1 -1]
//   [-1  8 -1]
//   [-1 -1 -1]
__global__ void laplacian3x3_u8_to_u8(const unsigned char* __restrict__ gray,
                                       unsigned char* __restrict__ out,
                                       int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int acc = 0;

    acc += -(int)gray[IDX(clampi(x - 1, 0, w - 1), clampi(y - 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x    , 0, w - 1), clampi(y - 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x + 1, 0, w - 1), clampi(y - 1, 0, h - 1), w)];

    acc += -(int)gray[IDX(clampi(x - 1, 0, w - 1), y, w)];
    acc +=  8 * (int)gray[IDX(x, y, w)];
    acc += -(int)gray[IDX(clampi(x + 1, 0, w - 1), y, w)];

    acc += -(int)gray[IDX(clampi(x - 1, 0, w - 1), clampi(y + 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x    , 0, w - 1), clampi(y + 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x + 1, 0, w - 1), clampi(y + 1, 0, h - 1), w)];

    int v = acc >= 0 ? acc : -acc;  // abs
    if (v > 255) v = 255;

    out[IDX(x, y, w)] = (unsigned char)v;
}

// General NxN Laplacian of Gaussian (LoG) convolution
// gray (uint8) * K (NxN float) -> out (float)
__global__ void conv_log_u8_to_f(const unsigned char* __restrict__ gray,
                                 const float* __restrict__ K, int N,
                                 float* __restrict__ out,
                                 int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int ky = -r; ky <= r; ++ky){
        int yy = clampi(y + ky, 0, h - 1);
        int krow = (ky + r) * N;
        for (int kx = -r; kx <= r; ++kx){
            int xx = clampi(x + kx, 0, w - 1);
            acc += (float)gray[IDX(xx, yy, w)] * K[krow + (kx + r)];
        }
    }

    out[IDX(x, y, w)] = acc;
}

// Convert float response to uint8 using abs() and clamp to [0,255]
__global__ void f_abs_to_u8(const float* __restrict__ in,
                             unsigned char* __restrict__ out,
                             int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float v = in[IDX(x, y, w)];
    if (v < 0.f) v = -v;
    if (v > 255.f) v = 255.f;

    out[IDX(x, y, w)] = (unsigned char)(v + 0.5f);
}

} // extern C
"""

# Compiled module (lazy)
_laplacian_mod = None
_laplacian_compiled = False

def _ensure_laplacian_compiled():
    """Compile Laplacian kernels if not already compiled."""
    global _laplacian_mod, _laplacian_compiled
    if not _laplacian_compiled:
        # Initialize CUDA first (shared context)
        _initialize_cuda()
        from cuda_kernels import compile_cuda_kernel_to_ptx
        import pycuda.driver as drv
        # Compile using nvcc directly with auto-detection
        ptx_code = compile_cuda_kernel_to_ptx(LAPLACIAN_CUDA_SRC, arch=None)
        _laplacian_mod = drv.module_from_buffer(ptx_code.encode())
        _laplacian_compiled = True


def make_log_kernel(N: int) -> np.ndarray:
    """
    Build an NxN Laplacian of Gaussian kernel as float32.
    
    LoG(x,y) = -((r^2 - 2*sigma^2)/sigma^4) * exp(-r^2 / (2*sigma^2))
    where sigma = N / 6
    
    Args:
        N: Kernel size (must be odd)
    
    Returns:
        NxN float32 kernel with sum close to 0
    """
    c = N // 2
    sigma = N / 6.0
    s2 = sigma * sigma
    s4 = s2 * s2

    K = np.empty((N, N), dtype=np.float32)
    s = 0.0

    for yy in range(-c, c + 1):
        for xx in range(-c, c + 1):
            r2 = float(xx * xx + yy * yy)
            val = -((r2 - 2.0 * s2) / s4) * np.exp(-r2 / (2.0 * s2))
            K[yy + c, xx + c] = val
            s += float(val)

    # Subtract mean so kernel sum is close to zero
    corr = s / float(N * N)
    K -= np.float32(corr)
    return K.astype(np.float32)


def apply_laplacian_cuda(
    image: np.ndarray,
    block_dim: Tuple[int, int],
    grid_dim: Tuple[int, int],
    mask_size: int = 3,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply Laplacian or LoG filter using CUDA.
    
    Args:
        image: Grayscale image (float32), shape (H, W)
        block_dim: (blockX, blockY)
        grid_dim: Ignored, calculated automatically
        mask_size: Kernel size NxN (default 3, must be odd)
                   3 = classic 3x3 Laplacian
                   >3 = Laplacian of Gaussian (LoG)
    
    Returns:
        (result_image, timings_dict)
    """
    import pycuda.driver as cuda
    
    _ensure_laplacian_compiled()
    
    if image.ndim != 2:
        raise ValueError("Image must be 2D (grayscale)")
    
    if mask_size % 2 == 0:
        raise ValueError(f"mask_size must be odd, got {mask_size}")
    
    h, w = image.shape
    N = mask_size
    use_log = (N != 3)
    
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
    d_out = cuda.mem_alloc(bytesGray)
    
    # Copy to GPU
    cuda.memcpy_htod(d_gray, gray)
    
    # Get functions
    laplacian3x3 = _laplacian_mod.get_function("laplacian3x3_u8_to_u8")
    conv_log = _laplacian_mod.get_function("conv_log_u8_to_f")
    f_abs_to_u8 = _laplacian_mod.get_function("f_abs_to_u8")
    
    # Measure time
    start = cuda.Event()
    stop = cuda.Event()
    start.record()
    
    if not use_log:
        # Classic 3x3 Laplacian
        laplacian3x3(d_gray, d_out, np.int32(w), np.int32(h), block=block, grid=grid)
    else:
        # LoG NxN: build kernel, allocate temp buffer, run convolution + abs
        h_K = make_log_kernel(N)
        d_K = cuda.mem_alloc(N * N * 4)
        d_tmpF = cuda.mem_alloc(Npix * 4)
        
        cuda.memcpy_htod(d_K, h_K)
        
        conv_log(d_gray, d_K, np.int32(N), d_tmpF, np.int32(w), np.int32(h), block=block, grid=grid)
        f_abs_to_u8(d_tmpF, d_out, np.int32(w), np.int32(h), block=block, grid=grid)
    
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


# Legacy function for backward compatibility with generic kernel approach
def laplacian_kernel() -> np.ndarray:
    """
    3x3 Laplacian kernel for edge detection.
    
    Laplacian is a second derivative operator that detects regions
    of rapid intensity change (edges). Uses 4-neighbor mask:
    
    [ 0  1  0]
    [ 1 -4  1]
    [ 0  1  0]
    
    Returns:
        np.ndarray: Laplacian kernel of shape (3, 3)
    """
    k = np.array(
        [
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0],
        ],
        dtype=np.float32,
    )
    return k
