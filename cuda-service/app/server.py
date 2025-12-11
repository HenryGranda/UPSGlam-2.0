# ============================================================
#  FASTAPI SERVICE: UPSGlam 2.0 - CUDA FILTER ENGINE
# ============================================================

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

# ===== IMPORTAR FILTROS CUDA / CPU =====
from filters.boomerang.boomerang import simulate_boomerang
from filters.ups_logo.ups_logo import apply_ups_logo
from filters.laplacian.laplacian import apply_laplacian
from filters.gauss.gauss import apply_gaussian
from filters.prewitt.prewitt import apply_prewitt
from filters.blox_blur.blox_blur import apply_blox


# ============================================================
#  CONFIGURACIÓN FASTAPI
# ============================================================

app = FastAPI(
    title="UPSGlam 2.0 CUDA Filter Engine",
    description="Servicio de filtros GPU/CPU para UPSGlam 2.0",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LECTURA DE IMAGEN COMO OpenCV
# ============================================================

def read_image(upload_file: UploadFile):
    data = upload_file.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("❌ No se pudo decodificar la imagen enviada.")
    return img


# ============================================================
#  ENDPOINT PRINCIPAL
# ============================================================

@app.get("/")
def root():
    return {"status": "UP", "service": "UPSGlam 2.0 CUDA Engine"}


# ------------------------------------------------------------
# 1) BOOMERANG (CUDA)
# ------------------------------------------------------------
@app.post("/filter/boomerang")
async def filter_boomerang(file: UploadFile = File(...)):
    img = read_image(file)

    gif_bytes, *_ = simulate_boomerang(img)

    return Response(
        content=gif_bytes,
        media_type="image/gif"
    )


# ------------------------------------------------------------
# 2) UPS LOGO AURA (CUDA)
# ------------------------------------------------------------
@app.post("/filter/ups-logo")
async def filter_ups_logo(file: UploadFile = File(...)):
    img = read_image(file)

    _, output_path = apply_ups_logo(img)

    return {"png_path": output_path}


# ------------------------------------------------------------
# 3) LAPLACIAN (CPU / CUDA si ya lo implementaste)
# ------------------------------------------------------------
@app.post("/filter/laplacian")
async def filter_laplacian(file: UploadFile = File(...)):
    img = read_image(file)

    output = apply_laplacian(img)
    out_name = f"/tmp/laplacian.png"
    cv2.imwrite(out_name, output)

    return {"png_path": out_name}


# ------------------------------------------------------------
# 4) GAUSSIAN BLUR (CPU o CUDA)
# ------------------------------------------------------------
@app.post("/filter/gaussian")
async def filter_gauss(file: UploadFile = File(...)):
    img = read_image(file)

    output = apply_gaussian(img)
    out_name = f"/tmp/gauss.png"
    cv2.imwrite(out_name, output)

    return {"png_path": out_name}


# ------------------------------------------------------------
# 5) PREWITT (CPU o CUDA)
# ------------------------------------------------------------
@app.post("/filter/prewitt")
async def filter_prewitt(file: UploadFile = File(...)):
    img = read_image(file)

    output = apply_prewitt(img)
    out_name = f"/tmp/prewitt.png"
    cv2.imwrite(out_name, output)

    return {"png_path": out_name}


# ------------------------------------------------------------
# 6) BLOX BLUR / MOSAIC (CPU o CUDA)
# ------------------------------------------------------------
@app.post("/filter/blox")
async def filter_blox(file: UploadFile = File(...)):
    img = read_image(file)

    output = apply_blox(img)
    out_name = f"/tmp/blox.png"
    cv2.imwrite(out_name, output)

    return {"png_path": out_name}


# ============================================================
#  READY FOR DOCKER DEPLOY
# ============================================================

# dentro del contenedor:
# uvicorn app.server:app --host 0.0.0.0 --port 8000
