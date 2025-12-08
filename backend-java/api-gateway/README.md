# UPSGlam API Gateway

API Gateway para UPSGlam 2.0 - Punto de entrada único para todos los microservicios.

## Puerto
- **Gateway**: `http://localhost:8080`

## Microservicios
- **Post Service**: `http://localhost:8081`
- **CUDA Service**: `http://localhost:5000`

## Rutas

### Posts y Feed
```
GET  /api/feed                    → Post Service
GET  /api/posts/{id}              → Post Service
GET  /api/posts/user/{userId}     → Post Service
POST /api/posts                   → Post Service
DELETE /api/posts/{id}            → Post Service
PATCH /api/posts/{id}/caption     → Post Service
```

### Likes
```
GET    /api/posts/{id}/likes      → Post Service
POST   /api/posts/{id}/likes      → Post Service
DELETE /api/posts/{id}/likes      → Post Service
```

### Comentarios
```
GET    /api/posts/{id}/comments   → Post Service
POST   /api/posts/{id}/comments   → Post Service
DELETE /api/posts/{id}/comments/{commentId} → Post Service
GET    /api/users/{userId}/comments → Post Service
```

### Imágenes
```
POST /api/images/preview           → Post Service
POST /api/images/upload            → Post Service
```

### Filtros (CUDA)
```
POST /api/filters/apply            → CUDA Service
GET  /api/filters/list             → CUDA Service
```

### Health Checks
```
GET /health                        → Gateway Health
GET /api/health/posts              → Post Service Health
GET /api/health/cuda               → CUDA Service Health
GET /actuator/health               → Gateway Actuator
```

## Iniciar el Gateway

```powershell
cd backend-java/api-gateway
.\start-gateway.ps1
```

## Probar el Gateway

```powershell
.\test-gateway.ps1
```

## Configuración CORS

El gateway tiene CORS habilitado para todos los orígenes (`*`), métodos y headers.

## Arquitectura

```
                    ┌─────────────────┐
                    │   Mobile App    │
                    │    (Flutter)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   API Gateway   │
                    │   Port: 8080    │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
       ┌────────▼────────┐      ┌────────▼────────┐
       │  Post Service   │      │  CUDA Service   │
       │   Port: 8081    │      │   Port: 5000    │
       │   (WebFlux)     │      │   (Python)      │
       └─────────────────┘      └─────────────────┘
```
