"""
PyCUDA Mock Service - Sin GPU
=================================
Servidor mock que simula el servicio PyCUDA sin necesitar GPU NVIDIA.
Simplemente devuelve la misma imagen sin procesar para que el frontend pueda probar.

Puerto: 5000
Uso: python app.py
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

app = FastAPI(
    title="PyCUDA Mock Service",
    description="Servidor mock para testing sin GPU NVIDIA - Devuelve la misma imagen",
    version="1.0.0"
)

# CORS - Permitir todas las peticiones
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Filtros disponibles (solo para referencia)
AVAILABLE_FILTERS = [
    {
        "name": "gaussian",
        "displayName": "Gaussian Blur",
        "description": "Suavizado gaussiano (MOCK - devuelve imagen original)",
        "category": "blur"
    },
    {
        "name": "box_blur",
        "displayName": "Box Blur",
        "description": "Desenfoque de caja (MOCK - devuelve imagen original)",
        "category": "blur"
    },
    {
        "name": "prewitt",
        "displayName": "Prewitt Edge",
        "description": "Detección de bordes Prewitt (MOCK - devuelve imagen original)",
        "category": "edge"
    },
    {
        "name": "laplacian",
        "displayName": "Laplacian Edge",
        "description": "Detección de bordes Laplacian (MOCK - devuelve imagen original)",
        "category": "edge"
    },
    {
        "name": "ups_logo",
        "displayName": "UPS Logo",
        "description": "Overlay del logo UPS (MOCK - devuelve imagen original)",
        "category": "creative"
    },
    {
        "name": "ups_color",
        "displayName": "UPS Colors",
        "description": "Colores institucionales UPS (MOCK - devuelve imagen original)",
        "category": "creative"
    }
]

FILTER_NAMES = [f["name"] for f in AVAILABLE_FILTERS]


@app.get("/")
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "service": "PyCUDA Mock Service",
        "version": "1.0.0",
        "description": "Servidor mock para testing sin GPU - Devuelve la misma imagen",
        "status": "running",
        "mode": "MOCK (no GPU processing)",
        "endpoints": {
            "health": "/health",
            "filters": "/filters",
            "apply_filter": "/filters/{filter_name}"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "pycuda-mock",
        "timestamp": datetime.now().isoformat(),
        "mode": "MOCK",
        "gpu": "Not required (mock mode)"
    }


@app.get("/filters")
async def list_filters():
    """Lista todos los filtros disponibles (mock)"""
    return {
        "filters": AVAILABLE_FILTERS,
        "total": len(AVAILABLE_FILTERS),
        "note": "MOCK MODE - All filters return original image"
    }


@app.post("/filters/{filter_name}")
async def apply_filter(filter_name: str, request: Request):
    """
    Aplica un filtro a la imagen (MOCK - devuelve la misma imagen)
    
    Args:
        filter_name: Nombre del filtro (gaussian, box_blur, prewitt, laplacian, ups_logo, ups_color)
        request: Request con la imagen en el body
    
    Returns:
        La misma imagen sin procesar (simulando procesamiento)
    """
    
    # Validar que el filtro existe
    if filter_name not in FILTER_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filter: {filter_name}. Available filters: {', '.join(FILTER_NAMES)}"
        )
    
    try:
        # Leer los bytes de la imagen del request body
        image_bytes = await request.body()
        
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="No image data provided"
            )
        
        # Log para debug
        print(f"[MOCK] Received image: {len(image_bytes)} bytes")
        print(f"[MOCK] Filter requested: {filter_name}")
        print(f"[MOCK] Returning original image (no processing)")
        
        # MOCK: Simplemente devolver la misma imagen sin procesar
        # En el servicio real, aquí se aplicaría el filtro con CUDA
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={
                "X-Mock-Service": "true",
                "X-Filter-Applied": filter_name,
                "X-Processing-Time": "0ms",
                "X-Note": "MOCK - Original image returned"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting PyCUDA Mock Service")
    print("=" * 60)
    print(f"Mode:     MOCK (no GPU processing)")
    print(f"Port:     5000")
    print(f"Filters:  {len(AVAILABLE_FILTERS)} available")
    print(f"Note:     Returns original image without processing")
    print("=" * 60)
    print("")
    print("Available endpoints:")
    print("  GET  /                    - Service info")
    print("  GET  /health              - Health check")
    print("  GET  /filters             - List all filters")
    print("  POST /filters/{name}      - Apply filter (mock)")
    print("")
    print("Available filters:")
    for f in AVAILABLE_FILTERS:
        print(f"  - {f['name']:12} ({f['category']})")
    print("")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
