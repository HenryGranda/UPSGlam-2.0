# 🐳 Docker Quick Start

## Inicio Rápido (3 pasos)

### 1. Build y Deploy
```powershell
.\build-docker.ps1
```

### 2. Ejecutar Tests
```powershell
.\test-docker.ps1
```

### 3. Usar el API
```powershell
# Ver filtros disponibles
curl http://localhost:5000/filters

# Aplicar filtro
curl -X POST "http://localhost:5000/filters/gaussian" `
     -H "Content-Type: image/jpeg" `
     --data-binary "@tu_imagen.jpg" `
     -o "resultado.jpg"
```

---

## Comandos Útiles

```powershell
# Ver logs en tiempo real
docker logs -f cuda-lab-backend

# Ver estado GPU dentro del container
docker exec -it cuda-lab-backend nvidia-smi

# Reiniciar servicio
docker restart cuda-lab-backend

# Detener servicio
docker stop cuda-lab-backend

# Eliminar container
docker rm -f cuda-lab-backend

# Rebuild después de cambios
.\build-docker.ps1
```

---

## Filtros Disponibles

- **gaussian**: Suavizado fuerte
- **box_blur**: Suavizado rápido  
- **prewitt**: Detección de bordes direccional
- **laplacian**: Detección de bordes omnidireccional
- **ups_logo**: Logo UPS con efectos de aura
- **ups_color**: Tinte corporativo UPS
- **boomerang**: Efecto de rastro con bolas
- **cr7**: Máscara facial sobre rostros detectados

---

## Troubleshooting

**Error: GPU no detectada**
```powershell
# Verificar drivers
nvidia-smi

# Verificar Docker puede acceder GPU
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**Error: Puerto 5000 en uso**
```powershell
# Cambiar puerto en docker-compose.yml
ports:
  - "8080:5000"
```

**Ver documentación completa**: [DOCKER.md](DOCKER.md)
