# 🎨 CUDA Lab Backend - UPSGlam 2.0

## 📋 Descripción General

El **CUDA Lab Backend** es el microservicio de procesamiento de imágenes que utiliza **PyCUDA** para aplicar filtros y efectos a imágenes mediante aceleración por GPU NVIDIA. Implementado con **FastAPI**, este servicio proporciona endpoints RESTful de alto rendimiento para el procesamiento de imágenes en tiempo real.

---

## 🏗️ Arquitectura

### Stack Tecnológico
- **Framework**: FastAPI 0.122.0
- **GPU Computing**: PyCUDA 2025.1.2
- **Image Processing**: OpenCV 4.10.0, Pillow 12.0.0
- **Runtime**: Python 3.10+
- **CUDA**: NVIDIA CUDA Toolkit 12.x
- **Server**: Uvicorn 0.38.0 (ASGI)
- **Container**: Docker con soporte GPU

### Dependencias Principales
```python
fastapi==0.122.0
pycuda==2025.1.2
opencv-python-headless==4.10.0.84
pillow==12.0.0
numpy==2.2.1
scipy==1.14.1
uvicorn==0.38.0
```

---

## 🚀 Características Principales

### 1. **Procesamiento GPU Acelerado**
- Convoluciones CUDA optimizadas
- Kernels personalizados para cada filtro
- Procesamiento paralelo masivo
- Performance 10-50x más rápido que CPU

### 2. **Filtros Implementados**

#### Filtros de Desenfoque (Blur)
- **Gaussian Blur**: Desenfoque gaussiano suave
- **Box Blur**: Desenfoque de caja uniforme

#### Filtros de Detección de Bordes
- **Prewitt**: Detección de bordes con operador Prewitt
- **Laplacian**: Detección de bordes con Laplaciano

#### Filtros Especiales
- **UPS Logo**: Overlay del logo de la Universidad
- **Boomerang**: Efecto espejo boomerang
- **CR7**: Máscara temática de CR7

### 3. **API Endpoints**

#### Endpoints Modernos (`/filters/*`)
- `POST /filters/apply`: Aplicar filtro por nombre
- `GET /filters/available`: Listar filtros disponibles

#### Endpoints Legacy (`/convolve`)
- `POST /convolve`: Endpoint legacy con config manual CUDA

---

## 🗂️ Estructura del Proyecto

```
cuda-lab-back/
├── app.py                    # FastAPI application
├── convolution_service.py    # CUDA convolution logic
├── cuda_kernels.py           # CUDA kernel definitions
├── image_utils.py            # Image processing utilities
├── progressive_convolution.py # Progressive processing
├── filters/                  # Filter implementations
│   ├── __init__.py
│   ├── gaussian.py
│   ├── box_blur.py
│   ├── prewitt.py
│   ├── laplacian.py
│   ├── ups_logo.py
│   ├── boomerang.py
│   ├── cr7.py
│   └── assets/              # Filter assets (logos, masks)
├── tests/                    # Unit tests
│   ├── test_filters.py
│   └── test_cuda.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔧 Configuración

### Requisitos del Sistema

#### Hardware
- **GPU**: NVIDIA GPU con Compute Capability 3.5+
- **VRAM**: Mínimo 2GB recomendado
- **Drivers**: NVIDIA Driver 525+ para CUDA 12

#### Software
- **CUDA Toolkit**: 12.x
- **Python**: 3.10+
- **pip**: 23.0+

### Variables de Entorno

| Variable | Descripción | Default | Requerido |
|----------|-------------|---------|-----------|
| `CUDA_VISIBLE_DEVICES` | GPUs a usar | `0` | ❌ |
| `PORT` | Puerto del servicio | `5000` | ❌ |
| `HOST` | Host binding | `0.0.0.0` | ❌ |
| `WORKERS` | Workers de Uvicorn | `1` | ❌ |

### Instalación Local

```bash
# 1. Clonar repositorio
cd backend-java/cuda-lab-back

# 2. Crear entorno virtual
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar CUDA
python -c "import pycuda.driver as cuda; cuda.init(); print(f'CUDA devices: {cuda.Device.count()}')"
```

---

## 📡 API Endpoints

### Base URL
- **Local**: `http://localhost:5000`
- **Docker**: `http://cuda-backend:5000`
- **Gateway**: `http://localhost:8080/api/filters`

---

### 1. Health Check

#### `GET /health`
Verificar estado del servicio y GPU.

**Response:**
```json
{
  "status": "ok",
  "cuda_available": true,
  "gpu_count": 1,
  "gpu_name": "NVIDIA GeForce RTX 3060"
}
```

---

### 2. Filtros Modernos

#### `GET /filters/available`
Obtener lista de filtros disponibles.

**Response:**
```json
{
  "filters": [
    {
      "name": "gaussian",
      "display_name": "Gaussian Blur",
      "description": "Suaviza la imagen con desenfoque gaussiano",
      "category": "blur",
      "parameters": {
        "mask_size": [3, 5, 7, 9],
        "default_mask_size": 5
      }
    },
    {
      "name": "prewitt",
      "display_name": "Prewitt Edge Detection",
      "description": "Detecta bordes usando operador Prewitt",
      "category": "edge_detection",
      "parameters": {
        "gain": "float",
        "default_gain": 8.0
      }
    }
    // ... más filtros
  ]
}
```

#### `POST /filters/apply`
Aplicar filtro a una imagen.

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:5000/filters/apply \
  -F "image=@photo.jpg" \
  -F "filter_name=gaussian" \
  -F "mask_size=5"
```

**Form Data:**
- `image`: Archivo de imagen (JPEG, PNG)
- `filter_name`: Nombre del filtro
- `mask_size` (opcional): Tamaño del kernel (3, 5, 7, 9)
- `gain` (opcional): Ganancia para filtros de bordes

**Response:**
- Content-Type: `image/jpeg` o `image/png`
- Body: Imagen procesada

**Error Response:**
```json
{
  "detail": "Filter 'invalid_filter' not found. Available filters: gaussian, box_blur, prewitt, ..."
}
```

---

### 3. Endpoint Legacy

#### `POST /convolve`
Aplicar convolución con configuración CUDA manual.

**Request:**
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "filter": {
    "type": "blur",
    "mask_size": 5,
    "gain": 8.0
  },
  "cuda_config": {
    "block_dim": [16, 16],
    "grid_dim": [32, 32]
  }
}
```

**Response:**
```json
{
  "result_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "execution_time_ms": 45.6,
  "filter_applied": "blur",
  "image_size": [1920, 1080]
}
```

---

## 🎨 Filtros Disponibles

### 1. Gaussian Blur (`gaussian`)

**Descripción**: Desenfoque suave usando distribución gaussiana.

**Parámetros**:
- `mask_size`: 3, 5, 7, 9 (default: 5)

**Uso**:
```bash
curl -X POST http://localhost:5000/filters/apply \
  -F "image=@photo.jpg" \
  -F "filter_name=gaussian" \
  -F "mask_size=5" \
  -o gaussian_output.jpg
```

**Kernel Gaussiano (5x5)**:
```
[ 1,  4,  7,  4, 1 ]
[ 4, 16, 26, 16, 4 ]
[ 7, 26, 41, 26, 7 ]
[ 4, 16, 26, 16, 4 ]
[ 1,  4,  7,  4, 1 ]
/ 273
```

---

### 2. Box Blur (`box_blur`)

**Descripción**: Desenfoque uniforme promediando píxeles vecinos.

**Parámetros**:
- `mask_size`: 3, 5, 7, 9 (default: 5)

**Uso**:
```bash
curl -X POST http://localhost:5000/filters/apply \
  -F "image=@photo.jpg" \
  -F "filter_name=box_blur" \
  -o box_blur_output.jpg
```

**Kernel Box (5x5)**:
```
[ 1, 1, 1, 1, 1 ]
[ 1, 1, 1, 1, 1 ]
[ 1, 1, 1, 1, 1 ]
[ 1, 1, 1, 1, 1 ]
[ 1, 1, 1, 1, 1 ]
/ 25
```

---

### 3. Prewitt Edge Detection (`prewitt`)

**Descripción**: Detecta bordes horizontales y verticales.

**Parámetros**:
- `gain`: Factor de amplificación (default: 8.0)

**Uso**:
```bash
curl -X POST http://localhost:5000/filters/apply \
  -F "image=@photo.jpg" \
  -F "filter_name=prewitt" \
  -F "gain=8.0" \
  -o prewitt_output.jpg
```

**Kernels Prewitt**:
```
Gx (horizontal):        Gy (vertical):
[-1, 0, 1]              [-1,-1,-1]
[-1, 0, 1]              [ 0, 0, 0]
[-1, 0, 1]              [ 1, 1, 1]
```

**Magnitude**: `sqrt(Gx² + Gy²) * gain`

---

### 4. Laplacian Edge Detection (`laplacian`)

**Descripción**: Detecta bordes usando operador Laplaciano.

**Parámetros**:
- `gain`: Factor de amplificación (default: 8.0)

**Uso**:
```bash
curl -X POST http://localhost:5000/filters/apply \
  -F "image=@photo.jpg" \
  -F "filter_name=laplacian" \
  -o laplacian_output.jpg
```

**Kernel Laplaciano (3x3)**:
```
[ 0, -1,  0 ]
[-1,  4, -1 ]
[ 0, -1,  0 ]
```

---

### 5. UPS Logo (`ups_logo`)

**Descripción**: Overlay del logo de la Universidad Politécnica Salesiana.

**Parámetros**: Ninguno

**Uso**:
```bash
curl -X POST http://localhost:5000/filters/apply \
  -F "image=@photo.jpg" \
  -F "filter_name=ups_logo" \
  -o ups_logo_output.jpg
```

**Características**:
- Logo PNG con transparencia
- Posicionamiento: Esquina inferior derecha
- Tamaño: 15% del ancho de imagen
- Alpha blending para transparencia

---

### 6. Boomerang (`boomerang`)

**Descripción**: Efecto espejo boomerang horizontal.

**Parámetros**: Ninguno

**Uso**:
```bash
curl -X POST http://localhost:5000/filters/apply \
  -F "image=@photo.jpg" \
  -F "filter_name=boomerang" \
  -o boomerang_output.jpg
```

**Efecto**: Duplica la imagen y crea un espejo horizontal.

---

### 7. CR7 Mask (`cr7`)

**Descripción**: Overlay de máscara temática CR7.

**Parámetros**: Ninguno

**Características**:
- Máscara PNG temática
- Alpha blending
- Posicionamiento centrado

---

## 🛠️ Desarrollo Local

### Ejecución del Servidor

```bash
# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Método 1: Uvicorn directo
uvicorn app:app --host 0.0.0.0 --port 5000 --reload

# Método 2: Python
python app.py

# Método 3: Con workers múltiples (producción)
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 4
```

### Testing

```bash
# Test health endpoint
curl http://localhost:5000/health

# Test con imagen
curl -X POST http://localhost:5000/filters/apply \
  -F "image=@test_image.jpg" \
  -F "filter_name=gaussian" \
  -o output.jpg

# Ver imagen resultante
open output.jpg  # Mac
xdg-open output.jpg  # Linux
start output.jpg  # Windows
```

### Python Testing Script

```python
# test_filters.py
import requests

def test_gaussian_filter():
    url = "http://localhost:5000/filters/apply"
    
    with open("test_image.jpg", "rb") as img:
        files = {"image": img}
        data = {"filter_name": "gaussian", "mask_size": 5}
        
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            with open("output_gaussian.jpg", "wb") as out:
                out.write(response.content)
            print("✅ Filtro aplicado correctamente")
        else:
            print(f"❌ Error: {response.json()}")

if __name__ == "__main__":
    test_gaussian_filter()
```

---

## 🐳 Docker

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
```

### Build & Run

```bash
# Build image
docker build -t upsglam-cuda-backend:latest .

# Run with GPU support
docker run -d \
  --name cuda-backend \
  -p 5000:5000 \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  upsglam-cuda-backend:latest

# Logs
docker logs -f cuda-backend

# GPU monitoring
nvidia-smi -l 1
```

### Docker Compose

```yaml
cuda-backend:
  build:
    context: ./cuda-lab-back
    dockerfile: Dockerfile
  container_name: upsglam-cuda-backend
  ports:
    - "5000:5000"
  environment:
    - CUDA_VISIBLE_DEVICES=0
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  networks:
    - upsglam-network
```

---

## 📊 Rendimiento

### Benchmarks

| Filtro | Resolución | CPU Time | GPU Time | Speedup |
|--------|------------|----------|----------|---------|
| Gaussian 5x5 | 1920x1080 | 450ms | 12ms | 37.5x |
| Box Blur 5x5 | 1920x1080 | 380ms | 9ms | 42.2x |
| Prewitt | 1920x1080 | 520ms | 15ms | 34.7x |
| Laplacian | 1920x1080 | 490ms | 13ms | 37.7x |

### Optimizaciones CUDA

```python
# Block dimensions optimizadas
BLOCK_SIZE = (16, 16)  # 256 threads per block

# Grid dimensions calculadas automáticamente
grid_x = (width + BLOCK_SIZE[0] - 1) // BLOCK_SIZE[0]
grid_y = (height + BLOCK_SIZE[1] - 1) // BLOCK_SIZE[1]
```

---

## 🐛 Troubleshooting

### Error: CUDA not available

```bash
# Verificar NVIDIA driver
nvidia-smi

# Verificar CUDA
python -c "import pycuda.driver as cuda; cuda.init(); print(cuda.Device.count())"

# Reinstalar PyCUDA
pip uninstall pycuda
pip install pycuda
```

### Error: Out of GPU memory

```python
# Reducir resolución de imagen antes de procesar
# O procesar por chunks (progressive_convolution.py)
```

### Error: Import errors

```bash
# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

---

## 📚 Referencias

- [PyCUDA Documentation](https://documen.tician.de/pycuda/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 📝 Changelog

### Version 1.0.0
- ✅ FastAPI REST API
- ✅ PyCUDA GPU acceleration
- ✅ 7 filtros implementados
- ✅ Endpoints modernos y legacy
- ✅ Docker con GPU support
- ✅ Health checks y monitoring

---

## 👥 Autor

UPSGlam Development Team - Universidad Politécnica Salesiana
