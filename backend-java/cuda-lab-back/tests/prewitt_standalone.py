# prewitt_standalone.py
# Versión standalone del Prewitt que funciona, para integrar al backend

import numpy as np
from PIL import Image
import io
import base64

# NO importar pycuda.autoinit aquí - se inicializará lazy
_cuda_initialized = False
_cuda_context = None

# =============================================================================
#  Código CUDA del filtro de Prewitt separable
# =============================================================================
KERNEL_SRC = r"""
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ size_t IDX(int x, int y, int w){
    return (size_t)y * w + x;
}

// Suma vertical tipo "box filter"
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

// Cálculo de gx a partir de V (suma vertical)
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

// Suma horizontal tipo "box filter"
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

// Cálculo de gy a partir de H (suma horizontal)
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

// Combinación final |gx| + |gy| y normalización a [0,255] en GRAYSCALE
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

# Compilar módulo CUDA (solo una vez)
_mod = None
_kernels_compiled = False

def _initialize_cuda():
    """Inicializa CUDA de forma lazy."""
    global _cuda_initialized, _cuda_context
    
    if _cuda_initialized:
        return
    
    try:
        import pycuda.driver as drv
        drv.init()
        # Crear contexto explícitamente
        _cuda_context = drv.Device(0).make_context()
        _cuda_initialized = True
        
        # Registrar cleanup al cerrar
        import atexit
        atexit.register(_cleanup_cuda)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize CUDA: {e}")

def _cleanup_cuda():
    """Limpia el contexto CUDA al cerrar."""
    global _cuda_context
    if _cuda_context is not None:
        try:
            _cuda_context.pop()
            _cuda_context = None
        except:
            pass

def _ensure_kernels():
    global _mod, _kernels_compiled
    if not _kernels_compiled:
        # Inicializar CUDA primero
        _initialize_cuda()
        
        from pycuda.compiler import SourceModule
        _mod = SourceModule(KERNEL_SRC)
        _kernels_compiled = True

def apply_prewitt_filter(image_base64: str, block_dim: tuple, grid_dim: tuple, N: int = 3, gain: float = 8.0):
    """
    Aplica filtro Prewitt a una imagen base64.
    
    Args:
        image_base64: Imagen en base64 (con o sin prefijo data:image/...)
        block_dim: (blockX, blockY) para CUDA
        grid_dim: (gridX, gridY) para CUDA
        N: Tamaño del kernel (impar, default 3)
        gain: Factor de realce (default 8.0)
    
    Returns:
        tuple: (result_base64, execution_time_ms)
    """
    import pycuda.driver as cuda
    
    _ensure_kernels()
    
    # Decodificar imagen base64
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # Grayscale
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape
    
    print(f"DEBUG: Image size: {w}x{h}, dtype: {arr.dtype}, range: [{arr.min()}, {arr.max()}]")
    
    # Aplanar imagen - IMPORTANTE: mantener como uint8
    gray = np.ascontiguousarray(arr.reshape(-1))
    
    Npix = w * h
    bytesGray = Npix
    
    # Configuración CUDA - Recalcular grid basado en tamaño de imagen
    blockX, blockY = block_dim
    # Calcular grid correcto para cubrir toda la imagen
    gridX = (w + blockX - 1) // blockX
    gridY = (h + blockY - 1) // blockY
    
    block = (blockX, blockY, 1)
    grid = (gridX, gridY, 1)
    
    print(f"DEBUG: Block: {block}, Grid: {grid}")
    
    # Reserva de memoria en GPU
    d_gray = cuda.mem_alloc(bytesGray)
    d_V = cuda.mem_alloc(Npix * 4)  # float
    d_H = cuda.mem_alloc(Npix * 4)
    d_gx = cuda.mem_alloc(Npix * 4)
    d_gy = cuda.mem_alloc(Npix * 4)
    d_out = cuda.mem_alloc(bytesGray)
    
    # Copiar imagen al dispositivo
    cuda.memcpy_htod(d_gray, gray)
    
    # Obtener funciones
    boxV = _mod.get_function("box_vert_u8_to_f")
    gxF = _mod.get_function("prewitt_x_from_V")
    boxH = _mod.get_function("box_horiz_u8_to_f")
    gyF = _mod.get_function("prewitt_y_from_H")
    comb = _mod.get_function("combine_mag_to_gray")
    
    # Medición de tiempo
    start = cuda.Event()
    stop = cuda.Event()
    start.record()
    
    # gx a partir de suma vertical
    boxV(d_gray, d_V,
         np.int32(w), np.int32(h), np.int32(N),
         block=block, grid=grid)
    gxF(d_V, d_gx,
        np.int32(w), np.int32(h), np.int32(N),
        block=block, grid=grid)
    
    # gy a partir de suma horizontal
    boxH(d_gray, d_H,
         np.int32(w), np.int32(h), np.int32(N),
         block=block, grid=grid)
    gyF(d_H, d_gy,
        np.int32(w), np.int32(h), np.int32(N),
        block=block, grid=grid)
    
    # Combinación |gx| + |gy| y normalización
    comb(d_gx, d_gy, d_out,
         np.int32(w), np.int32(h),
         np.int32(N), np.float32(gain),
         block=block, grid=grid)
    
    stop.record()
    stop.synchronize()
    elapsed_ms = start.time_till(stop)
    
    # DEBUG: Copiar gradientes para ver valores
    gx_debug = np.empty(Npix, dtype=np.float32)
    gy_debug = np.empty(Npix, dtype=np.float32)
    cuda.memcpy_dtoh(gx_debug, d_gx)
    cuda.memcpy_dtoh(gy_debug, d_gy)
    print(f"DEBUG: gx range: [{gx_debug.min():.2f}, {gx_debug.max():.2f}]")
    print(f"DEBUG: gy range: [{gy_debug.min():.2f}, {gy_debug.max():.2f}]")
    mag_debug = np.abs(gx_debug) + np.abs(gy_debug)
    print(f"DEBUG: magnitude range before normalization: [{mag_debug.min():.2f}, {mag_debug.max():.2f}]")
    
    # Copiar resultado al host
    out = np.empty(bytesGray, dtype=np.uint8)
    cuda.memcpy_dtoh(out, d_out)
    
    print(f"DEBUG: Output range: [{out.min()}, {out.max()}]")
    print(f"DEBUG: Non-zero pixels: {np.count_nonzero(out)} / {out.size}")
    
    # Convertir a imagen
    out_img = out.reshape(h, w)
    img_pil = Image.fromarray(out_img, mode='L')
    
    # Convertir a base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    b64_bytes = base64.b64encode(buffer.read())
    b64_str = b64_bytes.decode("utf-8")
    result_base64 = f"data:image/png;base64,{b64_str}"
    
    return result_base64, elapsed_ms
