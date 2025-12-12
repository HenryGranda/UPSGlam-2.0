# backend/cuda_kernels.py
"""
Orquestador CUDA - solo maneja kernel genérico de convolución.
Cada filtro especializado (como Prewitt) se implementa en filters/
"""

from typing import Dict, Tuple
import numpy as np
import os
import subprocess
import tempfile

# Lazy imports de CUDA
CUDA_AVAILABLE = None
CUDA_ERROR = None
_cuda_initialized = False


def _detect_cuda_arch():
    """
    Detectar automáticamente la arquitectura CUDA de la GPU disponible.
    Returns:
        str: Arquitectura CUDA (ej: 'sm_75', 'sm_86', 'sm_89')
    """
    try:
        import pycuda.autoinit
        import pycuda.driver as drv
        
        device = drv.Device(0)
        compute_capability = device.compute_capability()
        arch = f"sm_{compute_capability[0]}{compute_capability[1]}"
        return arch
    except Exception as e:
        # Fallback a arquitecturas comunes si no se puede detectar
        print(f"Warning: Could not detect CUDA arch, trying common architectures: {e}")
        return None


def compile_cuda_kernel_to_ptx(kernel_source, arch=None):
    """
    Compilar kernel CUDA usando nvcc directamente (bypass PyCUDA auto-detection)
    Args:
        kernel_source (str): Código fuente del kernel CUDA
        arch (str): Arquitectura CUDA (None = auto-detect)
    Returns:
        str: Código PTX compilado
    """
    # Auto-detectar arquitectura si no se especifica
    if arch is None:
        arch = _detect_cuda_arch()
    
    # Crear archivo temporal para el código CUDA
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(kernel_source)
        cu_file = f.name
    
    # Compilar a PTX usando nvcc
    ptx_file = cu_file.replace('.cu', '.ptx')
    
    # Lista de arquitecturas a intentar (de más nueva a más vieja)
    architectures = [arch] if arch else ['sm_89', 'sm_86', 'sm_75', 'sm_70', 'sm_61', 'sm_52']
    
    last_error = None
    for try_arch in architectures:
        if try_arch is None:
            continue
            
        cmd = ['nvcc', f'-arch={try_arch}', '--ptx', cu_file, '-o', ptx_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Compilación exitosa
            with open(ptx_file, 'r') as f:
                ptx_code = f.read()
            
            # Limpiar archivos temporales
            os.unlink(cu_file)
            os.unlink(ptx_file)
            
            print(f"CUDA kernel compiled successfully for {try_arch}")
            return ptx_code
        else:
            last_error = result.stderr
            # Continuar con la siguiente arquitectura
    
    # Si ninguna arquitectura funcionó, limpiar y lanzar error
    try:
        os.unlink(cu_file)
        if os.path.exists(ptx_file):
            os.unlink(ptx_file)
    except:
        pass
    
    raise Exception(f"CUDA compilation failed for all architectures. Last error: {last_error}")


# Generic 2D convolution kernel
CUDA_KERNEL_SRC = r"""
extern "C" {

__global__ void convolution(
    const float* img,
    const float* kernel,
    float* out,
    int width,
    int height,
    int ksize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int r = ksize / 2;
    float acc = 0.0f;

    for (int ky = -r; ky <= r; ky++) {
        for (int kx = -r; kx <= r; kx++) {
            int ix = x + kx;
            int iy = y + ky;

            // border clamping
            if (ix < 0) ix = 0;
            if (ix >= width) ix = width - 1;
            if (iy < 0) iy = 0;
            if (iy >= height) iy = height - 1;

            float img_val = img[iy * width + ix];
            float k_val = kernel[(ky + r) * ksize + (kx + r)];

            acc += img_val * k_val;
        }
    }

    out[y * width + x] = acc;
}

} // extern C
"""

# Lazy compiled module
_mod = None
_convolution_kernel = None


def _initialize_cuda():
    """Initialize CUDA in lazy mode."""
    global CUDA_AVAILABLE, CUDA_ERROR, _cuda_initialized
    
    if _cuda_initialized:
        return
    
    try:
        import pycuda.driver as drv
        drv.init()
        global _cuda_context
        _cuda_context = drv.Device(0).make_context()
        CUDA_AVAILABLE = True
        _cuda_initialized = True
    except Exception as e:
        CUDA_AVAILABLE = False
        CUDA_ERROR = str(e)
        _cuda_initialized = True


def _ensure_cuda_compiled():
    """Compile CUDA kernel if not already compiled."""
    global _mod, _convolution_kernel
    
    _initialize_cuda()
    
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            f"CUDA is not available. Error during initialization: {CUDA_ERROR}. "
            "Make sure you have: 1) NVIDIA GPU, 2) CUDA drivers, 3) CUDA toolkit, 4) Visual Studio with cl.exe"
        )
    
    if _mod is None:
        try:
            from pycuda.compiler import SourceModule
            # Forzar arquitectura sm_89 para RTX 5070 Ti Blackwell
            _mod = SourceModule(
                CUDA_KERNEL_SRC,
                options=["-arch=sm_89"]
            )
            _convolution_kernel = _mod.get_function("convolution")
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile CUDA kernel. Make sure CUDA toolkit and Visual Studio are installed. Error: {e}"
            )


def convolve_gpu_single(
    image: np.ndarray,
    kernel: np.ndarray,
    block_dim: Tuple[int, int],
    grid_dim: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply generic 2D convolution on GPU.
    Used by: Box Blur, Gaussian, Laplacian
    """
    import pycuda.driver as cuda
    
    _ensure_cuda_compiled()

    if image.ndim != 2:
        raise ValueError("image must be 2D (grayscale)")

    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a 2D square matrix")

    height, width = image.shape
    ksize = kernel.shape[0]

    # Ensure contiguous float32
    img_f = np.ascontiguousarray(image.astype(np.float32))
    ker_f = np.ascontiguousarray(kernel.astype(np.float32))

    # Allocate device memory
    img_gpu = cuda.mem_alloc(img_f.nbytes)
    ker_gpu = cuda.mem_alloc(ker_f.nbytes)
    out_gpu = cuda.mem_alloc(img_f.nbytes)

    # Copy host -> device
    cuda.memcpy_htod(img_gpu, img_f)
    cuda.memcpy_htod(ker_gpu, ker_f)

    # Configuration
    block = (int(block_dim[0]), int(block_dim[1]), 1)
    grid = (int(grid_dim[0]), int(grid_dim[1]), 1)

    # Events for timing
    start_evt = cuda.Event()
    end_evt = cuda.Event()

    start_evt.record()
    _convolution_kernel(
        img_gpu,
        ker_gpu,
        out_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(ksize),
        block=block,
        grid=grid,
    )
    end_evt.record()
    end_evt.synchronize()

    kernel_time_ms = start_evt.time_till(end_evt)

    # Copy result device -> host
    out_host = np.empty_like(img_f)
    cuda.memcpy_dtoh(out_host, out_gpu)

    timings: Dict[str, float] = {
        "execution_time_ms": float(kernel_time_ms),
        "kernel_time_ms": float(kernel_time_ms),
    }
    return out_host.astype(np.float32), timings
