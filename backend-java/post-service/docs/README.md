# 📸 Post Service - Documentación y Tests

Este directorio contiene toda la documentación y scripts de prueba para el **Post Service** de UPSGlam.

## 📁 Archivos

- **`doc-post-service.md`**: Documentación completa con todos los endpoints, ejemplos y troubleshooting
- **`start-post.ps1`**: Script para iniciar el servicio con todas las variables de entorno configuradas
- **`test-post-flow.ps1`**: Script automatizado que prueba todos los endpoints (14 tests)

## 🚀 Inicio Rápido

### 1. Iniciar el servicio

```powershell
cd docs
.\start-post.ps1
```

Espera a ver:
```
Netty started on port 8081
Started PostServiceApplication
```

### 2. Ejecutar tests automáticos

**IMPORTANTE**: Primero necesitas un usuario registrado en auth-service:

```powershell
# Terminal 1: Iniciar auth-service
cd backend-java/auth-service/docs
.\start-auth.ps1

# Terminal 2: Registrar usuario de prueba
$body = @{
    email = "testpost@ups.edu.ec"
    password = "test123456"
    username = "testpost"
    fullName = "Test Post User"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8082/api/auth/register" -Method POST -ContentType "application/json" -Body $body
```

Luego ejecuta los tests:

```powershell
cd backend-java/post-service/docs
.\test-post-flow.ps1
```

## 📋 Tests Incluidos

El script `test-post-flow.ps1` ejecuta **14 tests completos**:

1. ✅ Health check del servicio
2. ✅ Crear imagen de prueba (PNG temporal)
3. ✅ Subir imagen a Supabase Storage
4. ✅ Crear post con imagen
5. ✅ Obtener feed de posts
6. ✅ Dar like al post
7. ✅ Crear comentario
8. ✅ Obtener post por ID
9. ✅ Obtener comentarios del post
10. ✅ Obtener likes del post
11. ✅ Actualizar caption del post
12. ✅ Eliminar comentario
13. ✅ Quitar like
14. ✅ Eliminar post

**Duración**: ~20 segundos

## 🏗️ Arquitectura

```
Post Service (Puerto 8081)
    ↓
┌──────────────────┬──────────────────┐
│  Firestore       │  Supabase        │
│  (Firebase)      │  Storage         │
├──────────────────┼──────────────────┤
│  • Posts         │  • Imágenes      │
│    metadata      │    finales       │
│  • Likes         │  • CDN público   │
│  • Comments      │  • URLs          │
└──────────────────┴──────────────────┘
```

## 🔑 Variables de Entorno

El script `start-post.ps1` configura automáticamente:

```powershell
# Firebase (Firestore)
$env:FIREBASE_API_KEY = "AIzaSyBYcnFxABxm3eyFpCD-nioQbZV1-NDzA5A"
$env:FIREBASE_PROJECT_ID = "upsglam-8c88f"

# Supabase (Storage)
$env:SUPABASE_URL = "https://opohishcukgkrkfdsgoa.supabase.co"
$env:SUPABASE_KEY = "eyJhbGci..."  # Anon key
$env:SUPABASE_SERVICE_ROLE_KEY = "eyJhbGci..."  # Service role key
```

## 📖 Endpoints Principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/images/upload` | Subir imagen a Supabase |
| `POST` | `/api/posts` | Crear post |
| `GET` | `/api/feed` | Obtener feed |
| `GET` | `/api/posts/{id}` | Obtener post por ID |
| `POST` | `/api/posts/{id}/likes` | Dar like |
| `POST` | `/api/posts/{id}/comments` | Crear comentario |
| `PATCH` | `/api/posts/{id}/caption` | Actualizar caption |
| `DELETE` | `/api/posts/{id}` | Eliminar post |

Ver `doc-post-service.md` para documentación completa.

## 🐛 Troubleshooting

### Error: "Usuario no encontrado"

Primero registra un usuario en auth-service:
```powershell
$body = @{ email="testpost@ups.edu.ec"; password="test123456"; username="testpost"; fullName="Test User" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8082/api/auth/register" -Method POST -ContentType "application/json" -Body $body
```

### Error: "Supabase API key not valid"

Verifica que las variables de entorno en `start-post.ps1` sean correctas.

### Error: "Firestore not available"

Verifica que `firebase-credentials.json` esté en `src/main/resources/`

## 📞 Soporte

Para más detalles, consulta `doc-post-service.md`.
