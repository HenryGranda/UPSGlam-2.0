# UPSGlam 2.0 - Docker Deployment Guide

## 📦 Arquitectura Dockerizada

Este proyecto implementa una arquitectura de microservicios completamente dockerizada con los siguientes servicios:

### Servicios

| Servicio | Puerto | Tecnología | Descripción |
|----------|--------|------------|-------------|
| **API Gateway** | 8080 | Spring Cloud Gateway | Punto de entrada único, enrutamiento |
| **Auth Service** | 8082 | Spring WebFlux + Firebase | Autenticación y usuarios |
| **Post Service** | 8081 | Spring WebFlux + Firebase + Supabase | Posts, comentarios, likes |
| **CUDA Backend** | 5000 | FastAPI + PyCUDA | Filtros de imagen con GPU |

### Red de Contenedores

Todos los servicios se ejecutan en la red `upsglam-network` (bridge), permitiendo comunicación directa entre contenedores usando sus nombres.

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway (8080)                       │
│              http://localhost:8080/api/**                    │
└────────────┬──────────────┬─────────────┬───────────────────┘
             │              │             │
             │              │             │
    ┌────────▼───────┐ ┌───▼──────┐ ┌───▼──────────────┐
    │ Auth Service   │ │ Post     │ │ CUDA Backend     │
    │ (8082)         │ │ Service  │ │ (5000)           │
    │ Firebase Auth  │ │ (8081)   │ │ PyCUDA + NVIDIA  │
    │ + Firestore    │ │ Firebase │ │ GPU Filters      │
    └────────────────┘ │ Supabase │ └──────────────────┘
                       └──────────┘
```

## 🚀 Inicio Rápido

### Prerrequisitos

1. **Docker Desktop** instalado con soporte GPU (NVIDIA)
2. **Docker Compose** v3.8+
3. **Credenciales Firebase** (archivo JSON)
4. **Credenciales Supabase** (URL + Service Key)

### Paso 1: Configurar Variables de Entorno

```powershell
# Copiar archivo de ejemplo
cd C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java
cp .env.example .env

# Editar .env con tus credenciales
notepad .env
```

Contenido de `.env`:
```env
FIREBASE_PROJECT_ID=tu-proyecto-id
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_SERVICE_KEY=tu-service-role-key
CUDA_VISIBLE_DEVICES=0
```

### Paso 2: Agregar Credenciales Firebase

Descarga tu archivo de credenciales Firebase y colócalo en:
```
C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\firebase-credentials.json
```

### Paso 3: Construir e Iniciar Todos los Servicios

```powershell
# Opción 1: Usar script automatizado
.\start-all-services.ps1

# Opción 2: Comandos manuales
docker-compose build
docker-compose up -d
```

### Paso 4: Verificar Estado de los Servicios

```powershell
# Ver estado de contenedores
docker-compose ps

# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f api-gateway
docker-compose logs -f cuda-backend
```

## 🔍 Health Checks

| Servicio | URL |
|----------|-----|
| API Gateway | http://localhost:8080/actuator/health |
| Auth Service | http://localhost:8082/actuator/health |
| Post Service | http://localhost:8081/actuator/health |
| CUDA Backend | http://localhost:5000/health |

## 🛠️ Comandos Útiles

### Iniciar y Detener

```powershell
# Iniciar todos los servicios
docker-compose up -d

# Detener todos los servicios (mantiene contenedores)
docker-compose stop
# O usar el script
.\stop-all-services.ps1

# Detener y eliminar contenedores
docker-compose down

# Detener, eliminar contenedores y volúmenes
docker-compose down -v
```

### Reconstruir Servicios

```powershell
# Reconstruir todos los servicios
docker-compose build --no-cache

# Reconstruir un servicio específico
docker-compose build --no-cache auth-service

# Reconstruir y reiniciar
docker-compose up -d --build
```

### Logs y Debugging

```powershell
# Ver logs de todos los servicios
docker-compose logs -f

# Ver logs de un servicio
docker-compose logs -f cuda-backend

# Ver últimas 100 líneas
docker-compose logs --tail=100 api-gateway

# Ver logs desde un timestamp
docker-compose logs --since 2025-12-12T10:00:00
```

### Acceder a un Contenedor

```powershell
# Bash en un contenedor
docker exec -it upsglam-cuda-backend /bin/bash

# Ejecutar comando en contenedor
docker exec upsglam-api-gateway ps aux
```

### Monitoreo de Recursos

```powershell
# Ver uso de recursos
docker stats

# Ver solo contenedores de UPSGlam
docker stats $(docker ps --filter "name=upsglam" -q)
```

## 🔧 Configuraciones por Perfil

### Perfil Docker (Predeterminado)

Los servicios Spring Boot usan el perfil `docker` automáticamente cuando se ejecutan en contenedores. Este perfil:

- Usa nombres de servicio de Docker (`auth-service`, `post-service`, `cuda-backend`)
- Configura memoria optimizada para contenedores
- Habilita actuator endpoints para health checks

### Perfil Local (Desarrollo)

Para desarrollo local sin Docker, los servicios usan `localhost`:

```yaml
# application.yml (local)
uri: http://localhost:8082  # Auth Service
uri: http://localhost:8081  # Post Service
uri: http://localhost:5000  # CUDA Backend
```

### Perfil Docker (Producción)

```yaml
# application-docker.yml
uri: http://auth-service:8082  # Container name
uri: http://post-service:8081  # Container name
uri: http://cuda-backend:5000  # Container name
```

## 📝 Solución de Problemas

### Los servicios no se comunican

**Problema**: Gateway retorna 503 Service Unavailable

**Solución**:
```powershell
# Verificar que todos estén en la misma red
docker network inspect upsglam-network

# Debe mostrar todos los contenedores conectados
```

### CUDA Backend no arranca

**Problema**: GPU no detectada

**Solución**:
```powershell
# Verificar soporte GPU en Docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Verificar logs del contenedor
docker-compose logs cuda-backend
```

### Auth/Post Service no pueden acceder a Firebase

**Problema**: firebase-credentials.json no encontrado

**Solución**:
```powershell
# Verificar que el archivo existe
Test-Path .\firebase-credentials.json

# Verificar montaje en contenedor
docker exec upsglam-auth-service ls -la /app/firebase-credentials.json
```

### Out of Memory

**Problema**: Contenedores consumen mucha memoria

**Solución**:
```yaml
# Editar docker-compose.yml
services:
  auth-service:
    environment:
      - JAVA_OPTS=-Xmx256m -Xms128m  # Reducir memoria
```

## 🔐 Seguridad

### Archivos Sensibles (NO COMMIT)

```gitignore
.env
firebase-credentials.json
```

### Variables de Entorno Requeridas

- `FIREBASE_PROJECT_ID`: ID de proyecto Firebase
- `SUPABASE_URL`: URL de tu proyecto Supabase
- `SUPABASE_SERVICE_KEY`: Service role key (no anon key)

## 🚢 Despliegue en Producción

### Docker Compose para Producción

```yaml
# docker-compose.prod.yml
services:
  api-gateway:
    image: registry.example.com/upsglam-api-gateway:latest
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

### Build para Producción

```powershell
# Build con tag de versión
docker-compose build
docker tag upsglam-api-gateway:latest registry.example.com/upsglam-api-gateway:v1.0.0

# Push a registry
docker push registry.example.com/upsglam-api-gateway:v1.0.0
```

## 📊 Monitoreo y Observabilidad

### Actuator Endpoints

```powershell
# Gateway Routes
curl http://localhost:8080/actuator/gateway/routes | jq

# Health Check con Detalles
curl http://localhost:8080/actuator/health | jq
```

### Prometheus Metrics (Futuro)

```yaml
# Agregar a docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## 📚 Referencias

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Spring Cloud Gateway](https://spring.io/projects/spring-cloud-gateway)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [Firebase Admin SDK](https://firebase.google.com/docs/admin/setup)

## 📄 Licencia

UPSGlam 2.0 - Universidad Politécnica Salesiana
