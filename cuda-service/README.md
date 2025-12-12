# 🎨 CUDA Service - UPSGlam 2.0

## 📋 Descripción

Servicio alternativo de procesamiento de imágenes con CUDA para UPSGlam. Similar al `cuda-lab-back` pero con estructura de proyecto diferente.

## 🏗️ Stack Tecnológico

- **Python**: 3.10+
- **FastAPI**: Web framework
- **PyCUDA**: GPU acceleration
- **CUDA**: 12.x
- **Docker**: Containerization

## 📁 Estructura

```
cuda-service/
├── app/
│   └── server.py          # FastAPI server
├── filters/
│   ├── __init__.py
│   ├── gaussian.py
│   ├── box_blur.py
│   ├── prewitt.py
│   ├── laplacian.py
│   ├── ups_logo.py
│   ├── boomerang.py
│   └── cr7.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Local Development

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
uvicorn app.server:app --host 0.0.0.0 --port 5000 --reload
```

### Docker

```bash
# Build
docker build -t upsglam-cuda-service:latest .

# Run con GPU
docker run -d \
  --name cuda-service \
  -p 5000:5000 \
  --gpus all \
  upsglam-cuda-service:latest
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Apply Filter
```bash
POST /filters/apply
Content-Type: multipart/form-data

FormData:
- image: File
- filter_name: string
```

## 🎨 Filtros Disponibles

1. **gaussian** - Gaussian Blur
2. **box_blur** - Box Blur  
3. **prewitt** - Prewitt Edge Detection
4. **laplacian** - Laplacian Edge Detection
5. **ups_logo** - UPS Logo Overlay
6. **boomerang** - Boomerang Effect
7. **cr7** - CR7 Mask

## 🔧 Configuración

### Variables de Entorno

```bash
CUDA_VISIBLE_DEVICES=0  # GPU to use
PORT=5000               # Server port
HOST=0.0.0.0           # Bind address
```

## 📚 Diferencias con cuda-lab-back

| Aspecto | cuda-service | cuda-lab-back |
|---------|--------------|---------------|
| Ubicación | `/cuda-service` | `/backend-java/cuda-lab-back` |
| Estructura | App separada | Dentro de backend |
| Uso | Alternativa/Testing | Principal |

## 📖 Referencias

- [PyCUDA Documentation](https://documen.tician.de/pycuda/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Main CUDA Backend README](../backend-java/cuda-lab-back/README-DETAILED.md)

---

**UPSGlam Development Team**  
Universidad Politécnica Salesiana
