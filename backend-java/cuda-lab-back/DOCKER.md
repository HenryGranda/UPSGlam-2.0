# Docker Setup para CUDA Lab Backend

## 📋 Requisitos Previos

### 1. Docker Desktop con WSL2 (Windows)
- Instalar [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Habilitar WSL2 backend en Docker Desktop settings

### 2. NVIDIA Container Toolkit
```powershell
# En WSL2 (Ubuntu), instalar NVIDIA Container Toolkit
wsl

# Actualizar repositorios
sudo apt-get update

# Instalar toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. Verificar GPU en Docker
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 Build y Run

### Opción A: Script Automatizado (Recomendado)

```powershell
# Build, deploy y test en un solo comando
.\build-docker.ps1

# Luego ejecutar tests
.\test-docker.ps1
```

### Opción B: Docker Compose

```powershell
# Build y Run
docker-compose up -d --build

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

### Opción C: Docker Manual

```powershell
# Build image
docker build -t cuda-lab-back:latest .

# Run container
docker run -d `
  --name cuda-lab-backend `
  --gpus all `
  -p 5000:5000 `
  -e CUDA_VISIBLE_DEVICES=0 `
  --restart unless-stopped `
  cuda-lab-back:latest

# Ver logs
docker logs -f cuda-lab-backend

# Detener
docker stop cuda-lab-backend
docker rm cuda-lab-backend
```

---

## 🧪 Probar el Servicio

### 1. Ejecutar Suite de Tests
```powershell
.\test-docker.ps1
```

### 2. Health Check Manual
```powershell
curl http://localhost:5000/health
```

### 3. Listar Filtros Disponibles
```powershell
curl http://localhost:5000/filters
```

### 4. Probar Filtro Específico
```powershell
# Gaussian filter (usando curl)
curl -X POST "http://localhost:5000/filters/gaussian" `
     -H "Content-Type: image/jpeg" `
     --data-binary "@husky.jpg" `
     -o "output_gaussian.jpg"

# CR7 face mask filter
curl -X POST "http://localhost:5000/filters/cr7" `
     -H "Content-Type: image/jpeg" `
     --data-binary "@husky.jpg" `
     -o "output_cr7.jpg"

# Prewitt edge detection
curl -X POST "http://localhost:5000/filters/prewitt" `
     -H "Content-Type: image/jpeg" `
     --data-binary "@husky.jpg" `
     -o "output_prewitt.jpg"
```

### 5. Test con PowerShell
```powershell
$imageBytes = [System.IO.File]::ReadAllBytes(".\husky.jpg")

Invoke-RestMethod `
    -Uri "http://localhost:5000/filters/gaussian" `
    -Method POST `
    -ContentType "image/jpeg" `
    -Body $imageBytes `
    -OutFile "output_gaussian.jpg"
```

---

## 🔍 Debugging

### Ver logs en tiempo real
```powershell
docker-compose logs -f cuda-lab-back
```

### Entrar al container
```powershell
docker exec -it cuda-lab-backend bash
```

### Verificar GPU dentro del container
```powershell
docker exec -it cuda-lab-backend nvidia-smi
```

### Verificar versión CUDA
```powershell
docker exec -it cuda-lab-backend nvcc --version
```

---

## 🐛 Solución de Problemas

### Error: "could not select device driver"
- Instalar NVIDIA Container Toolkit
- Reiniciar Docker Desktop
- Verificar que WSL2 tenga acceso a GPU: `wsl nvidia-smi`

### Error: "failed to create shim task"
- Reiniciar Docker Desktop
- Verificar que Docker Desktop esté usando WSL2 backend

### Error: "CUDA initialization failed"
- Verificar drivers NVIDIA actualizados en Windows
- Reiniciar sistema
- Verificar `nvidia-smi` en WSL2

### Puerto 5000 en uso
```powershell
# Cambiar puerto en docker-compose.yml
ports:
  - "8080:5000"  # usar puerto 8080 en host
```

---

## 📊 Monitoreo

### Ver uso de GPU
```powershell
# Desde Windows (host)
nvidia-smi -l 1

# Desde container
docker exec -it cuda-lab-backend watch -n 1 nvidia-smi
```

### Ver recursos del container
```powershell
docker stats cuda-lab-backend
```

---

## 🔄 Rebuild Después de Cambios

```powershell
# Rebuild sin cache
docker-compose build --no-cache

# Rebuild y restart
docker-compose up -d --build
```

---

## 📝 Notas Importantes

1. **Imagen Base**: Usa `nvidia/cuda:12.0.0-runtime-ubuntu22.04`
   - Si tu GPU usa CUDA 11.x, cambiar a `nvidia/cuda:11.8.0-runtime-ubuntu22.04`

2. **GPU Access**: El flag `--gpus all` da acceso a todas las GPUs
   - Para GPU específica: `--gpus '"device=0"'`

3. **Performance**: Runtime es más ligero que devel
   - Si necesitas compilar código CUDA, usar `devel` en vez de `runtime`

4. **Volume Mounts**: Para desarrollo con hot-reload:
   ```yaml
   volumes:
     - ./:/app
   command: ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
   ```

---

## 🚢 Producción

Para producción, considera:

1. **Multi-stage build** para reducir tamaño
2. **Health checks** configurados
3. **Resource limits** en docker-compose
4. **Logging** a volume o servicio externo
5. **Secrets** para API keys (si aplica)

---

## 📦 Limpieza

```powershell
# Detener y eliminar containers
docker-compose down

# Eliminar imagen
docker rmi cuda-lab-back:latest

# Limpiar todo Docker
docker system prune -a --volumes
```
