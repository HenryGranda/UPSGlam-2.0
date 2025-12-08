# cuda-lab-back/app.py


from fastapi import FastAPI, HTTPException, Request, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List

from convolution_service import process_convolution_request, process_convolution_bytes
from image_utils import decode_image_base64

app = FastAPI(title="CUDA Image Lab Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ---------- Pydantic Models ----------

class FilterConfig(BaseModel):
    type: str           # filter type, e.g., "blur", "sharpen"
    mask_size: int      # filter mask size
    gain: float = 8.0   # gain for edge enhancement (Prewitt), default 8.0
    
class CudaConfig(BaseModel):
    block_dim: List[int]   # [blockDimX, blockDimY]
    grid_dim: List[int]    # [gridDimX, gridDimY]

class ConvolutionRequest(BaseModel):
    image_base64: str
    filter: FilterConfig
    cuda_config: CudaConfig


# ---------- Routes ----------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ==================== LEGACY ENDPOINTS (cuda-lab-back) ====================

@app.post("/convolve")
def convolve(req: ConvolutionRequest):
    """
    Main endpoint that applies convolution on the GPU.
    Receives a ConvolutionRequest, passes it to convolution_service, and returns the result.
    
    LEGACY: For cuda-lab-back compatibility. Uses base64 encoding.
    """
    try:
        payload = req.model_dump()  # dict with image_base64, filter, cuda_config
        result = process_convolution_request(payload)
        return result
    except ValueError as e:
        # Data validation errors (mask_size, filter, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # CUDA errors (no GPU, compilation failed, etc.)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Unexpected errors - show details for debugging
        import traceback
        error_detail = f"Internal server error: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


# ==================== UPSGLAM ENDPOINTS (bytes) ====================

@app.post("/filters/{filter_name}")
async def apply_filter(
    filter_name: str = Path(..., description="Filter name: prewitt, laplacian, gaussian, box_blur, ups_logo, ups_color"),
    request: Request = None
):
    """
    Apply filter to image and return filtered image bytes.
    
    UPSGlam endpoint: Receives raw image bytes (JPEG/PNG), returns filtered image bytes.
    
    Request:
        - Body: Raw image bytes (multipart or binary)
        - Content-Type: image/jpeg or image/png
    
    Response:
        - Body: Filtered image bytes
        - Content-Type: image/jpeg
    
    Filter configurations (preset):
        - gaussian: mask_size=121 (strong blur)
        - box_blur: mask_size=91 (strong smoothing)
        - prewitt: mask_size=3, gain=8.0 (edge detection)
        - laplacian: mask_size=3 (edge detection)
        - ups_logo: mask_size=5 (blur + UPS logo)
        - ups_color: mask_size=1 (UPS color tint)
    
    Example:
        curl -X POST "http://localhost:5000/filters/gaussian" \\
             -H "Content-Type: image/jpeg" \\
             --data-binary "@input.jpg" \\
             -o "output_gaussian.jpg"
    """
    try:
        # Read raw image bytes from request body
        image_bytes = await request.body()
        
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image body")
        
        # Process with filter (preset configurations)
        result_bytes = process_convolution_bytes(
            image_bytes=image_bytes,
            filter_name=filter_name,
            preserve_color=True    # RGB output
        )
        
        return Response(
            content=result_bytes,
            media_type="image/jpeg",
            headers={
                "X-Filter-Applied": filter_name,
                "Cache-Control": "no-cache"
            }
        )
        
    except ValueError as e:
        # Invalid filter name or image format
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # CUDA errors
        raise HTTPException(status_code=503, detail=f"GPU processing error: {e}")
    except Exception as e:
        import traceback
        error_detail = f"Filter processing error: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/filters")
async def list_filters():
    """
    List available filters with their preset configurations.
    
    UPSGlam endpoint: Returns catalog of available filters.
    """
    return {
        "filters": [
            {
                "name": "prewitt",
                "description": "Detección de bordes direccional (primera derivada)",
                "type": "convolución",
                "config": {"mask_size": 3, "gain": 8.0}
            },
            {
                "name": "laplacian",
                "description": "Detección de bordes omnidireccional (segunda derivada)",
                "type": "convolución",
                "config": {"mask_size": 3}
            },
            {
                "name": "gaussian",
                "description": "Suavizado con distribución gaussiana (fuerte)",
                "type": "convolución",
                "config": {"mask_size": 121}
            },
            {
                "name": "box_blur",
                "description": "Suavizado rápido con promedio simple (fuerte)",
                "type": "convolución",
                "config": {"mask_size": 91}
            },
            {
                "name": "ups_logo",
                "description": "Filtro creativo con logo de la UPS",
                "type": "creativo",
                "config": {"mask_size": 5}
            },
            {
                "name": "ups_color",
                "description": "Tinte con colores corporativos de la UPS",
                "type": "creativo",
                "config": {"mask_size": 1}
            }
        ]
    }
