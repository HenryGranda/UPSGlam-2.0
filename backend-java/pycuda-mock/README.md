# PyCUDA Mock Service

> 🎭 Servidor mock para testing sin GPU NVIDIA - Devuelve la misma imagen sin procesar

## 📋 Descripción

Este es un servidor **mock** del PyCUDA Service que **no requiere GPU NVIDIA**. Simplemente devuelve la imagen original sin aplicar ningún filtro, permitiendo que el equipo de frontend pueda desarrollar y probar la aplicación sin necesitar hardware especializado.

**Diferencias con el servicio real:**
- ❌ **NO procesa imágenes** - Devuelve la imagen original
- ❌ **NO usa GPU/CUDA** - No requiere drivers NVIDIA
- ✅ **API compatible** - Mismos endpoints y respuestas
- ✅ **Rápido** - Sin procesamiento, responde instantáneamente
- ✅ **Portable** - Corre en cualquier máquina con Python

## 🚀 Inicio Rápido

### Requisitos
- Python 3.7+
- pip

### Instalación y Ejecución

**Opción 1: Script automático (Windows)**
```powershell
cd backend-java/pycuda-mock
.\start-mock.ps1
```

**Opción 2: Manual**
```bash
# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor
python app.py
```

El servidor estará disponible en: `http://localhost:5000`

## 📡 Endpoints

### Health Check
```bash
GET http://localhost:5000/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "pycuda-mock",
  "timestamp": "2025-12-09T18:30:00",
  "mode": "MOCK",
  "gpu": "Not required (mock mode)"
}
```

### Listar Filtros
```bash
GET http://localhost:5000/filters
```

**Response:**
```json
{
  "filters": [
    {
      "name": "gaussian",
      "displayName": "Gaussian Blur",
      "description": "Suavizado gaussiano (MOCK - devuelve imagen original)",
      "category": "blur"
    },
    ...
  ],
  "total": 6,
  "note": "MOCK MODE - All filters return original image"
}
```

### Aplicar Filtro (Mock)
```bash
POST http://localhost:5000/filters/{filter_name}
Content-Type: image/jpeg
Body: [imagen JPEG]
```

**Filtros disponibles:**
- `gaussian` - Gaussian Blur
- `box_blur` - Box Blur
- `prewitt` - Prewitt Edge Detection
- `laplacian` - Laplacian Edge Detection
- `ups_logo` - UPS Logo Overlay
- `ups_color` - UPS Colors

**Response:**
- Body: La **misma imagen** sin procesar
- Headers:
  - `X-Mock-Service: true`
  - `X-Filter-Applied: {filter_name}`
  - `X-Note: MOCK - Original image returned`

## 🧪 Testing

### Con curl
```bash
# Health check
curl http://localhost:5000/health

# Listar filtros
curl http://localhost:5000/filters

# Aplicar filtro (mock)
curl -X POST "http://localhost:5000/filters/gaussian" \
  -H "Content-Type: image/jpeg" \
  --data-binary "@imagen.jpg" \
  -o "imagen_mock.jpg"
```

### Con PowerShell
```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:5000/health"

# Listar filtros
Invoke-RestMethod -Uri "http://localhost:5000/filters"

# Aplicar filtro (mock)
$image = [System.IO.File]::ReadAllBytes("imagen.jpg")
Invoke-RestMethod -Uri "http://localhost:5000/filters/gaussian" `
  -Method POST `
  -ContentType "image/jpeg" `
  -Body $image `
  -OutFile "imagen_mock.jpg"
```

## 🔧 Integración con Backend

El Post Service está configurado para conectarse automáticamente al puerto 5000. No necesitas cambiar nada en la configuración.

**Archivo:** `post-service/src/main/resources/application-local.yml`
```yaml
pycuda:
  service:
    url: "http://localhost:5000"  # Apunta al mock
    timeout: 30000
```

## ⚠️ Importante para Producción

**Este es un servicio MOCK solo para desarrollo/testing.**

Para producción, debes usar el servicio PyCUDA real:
- Ubicación: `backend-java/cuda-lab-back/`
- Requiere: GPU NVIDIA con CUDA
- Procesa: Imágenes con filtros reales

## 📝 Logs del Servidor

Cuando el mock recibe una petición, muestra:
```
[MOCK] Received image: 83516 bytes
[MOCK] Filter requested: gaussian
[MOCK] Returning original image (no processing)
```

Los headers de respuesta también indican que es un mock:
```
X-Mock-Service: true
X-Filter-Applied: gaussian
X-Processing-Time: 0ms
X-Note: MOCK - Original image returned
```

## 🎯 Casos de Uso

✅ **Desarrollo Frontend** - El equipo mobile puede desarrollar sin GPU  
✅ **Testing Rápido** - Probar flujo completo sin esperar procesamiento  
✅ **CI/CD** - Ejecutar tests automatizados sin hardware especial  
✅ **Demos** - Mostrar la aplicación sin necesitar GPU  

## 🔄 Diferencias con el Servicio Real

| Aspecto | Mock Service | Real Service |
|---------|-------------|--------------|
| Puerto | 5000 | 5000 |
| Endpoints | Iguales | Iguales |
| Procesamiento | ❌ No procesa | ✅ CUDA GPU |
| Velocidad | Instantáneo | 2-5 segundos |
| Requiere GPU | ❌ No | ✅ Sí (NVIDIA) |
| Calidad imagen | Original | Filtrada |
| Para | Desarrollo/Testing | Producción |

## 📚 Documentación Relacionada

- **API Documentation:** `mobile-app/API_DOCUMENTATION.md`
- **PyCUDA Real:** `backend-java/cuda-lab-back/QUICKSTART.md`
- **Post Service:** `backend-java/post-service/README.md`

---

**Desarrollado para UPSGlam 2.0**  
Universidad Politécnica Salesiana - Diciembre 2025
