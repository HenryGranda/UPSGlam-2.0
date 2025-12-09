# 📸 Post Service - Documentación Completa

## 📋 Descripción General

El **Post Service** es el microservicio responsable de la **gestión de publicaciones, imágenes y contenido social** en UPSGlam. Utiliza una **arquitectura híbrida** combinando Supabase Storage para imágenes y Firestore para metadata.

**Puerto**: `8081`  
**Base Path**: `/api`  
**Tecnología**: Spring WebFlux (Reactivo)

---

## ⚙️ Configuración Rápida

### **Iniciar el Servidor**

```powershell
cd backend-java/post-service/docs
.\start-post.ps1
```

El script `start-post.ps1` configura automáticamente:
```powershell
# Firebase (Firestore para metadata)
$env:FIREBASE_API_KEY = "AIzaSyBYcnFxABxm3eyFpCD-nioQbZV1-NDzA5A"
$env:FIREBASE_PROJECT_ID = "upsglam-8c88f"

# Supabase (Storage para imágenes)
$env:SUPABASE_URL = "https://opohishcukgkrkfdsgoa.supabase.co"
$env:SUPABASE_KEY = "eyJhbGci..."  # Anon key
$env:SUPABASE_SERVICE_ROLE_KEY = "eyJhbGci..."  # Service role key
```

### **Ejecutar Tests Automáticos**

```powershell
cd docs
.\test-post-flow.ps1
```

Este script ejecuta **tests completos** de todos los endpoints del servicio.

---

## 🏗️ Arquitectura Híbrida

```
Mobile App
    ↓
API Gateway (Post Service)
    ↓
┌──────────────────┬──────────────────┬─────────────────┐
│  Firestore       │  Supabase        │  PyCUDA         │
│  (Firebase)      │  Storage         │  Service        │
├──────────────────┼──────────────────┼─────────────────┤
│  • Posts         │  • Imágenes      │  • Filtros      │
│    metadata      │    finales       │    (GPU)        │
│  • Likes         │  • CDN público   │  • Logos UPS    │
│  • Comments      │  • URLs          │  • Efectos      │
│  • Timestamps    │                  │                 │
└──────────────────┴──────────────────┴─────────────────┘
```

### **¿Por qué esta arquitectura?**

✅ **Firestore (Firebase)**: Excelente para datos en tiempo real (likes, comments)  
✅ **Supabase Storage**: CDN rápido y económico para imágenes  
✅ **PyCUDA**: Procesamiento GPU para filtros en tiempo real

---

## 🚀 Endpoints Principales

### **1. Subir Imagen a Supabase**

Sube una imagen directamente a Supabase Storage y obtiene la URL pública.

#### **Endpoint**: `POST /api/images/upload`

**Headers**:
```http
Content-Type: multipart/form-data
Authorization: Bearer {firebase-id-token}
```

**Request Body** (multipart):
```
image: <binary-image-data>
```

**Response** (200 OK):
```json
{
  "url": "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/posts/user123-1733685464.jpg",
  "fileName": "user123-1733685464.jpg",
  "size": 245678,
  "contentType": "image/jpeg"
}
```

**Ejemplo PowerShell**:
```powershell
$headers = @{
    Authorization = "Bearer $TOKEN"
}

$form = @{
    image = Get-Item "C:\Users\...\foto.jpg"
}

$response = Invoke-RestMethod -Uri "http://localhost:8081/api/images/upload" `
    -Method POST -Headers $headers -Form $form

Write-Host "URL de imagen: $($response.url)"
```

---

### **2. Preview con Filtro (PyCUDA)**

Aplica un filtro CUDA a una imagen y retorna preview (sin guardar).

#### **Endpoint**: `POST /api/images/preview`

**Headers**:
```http
Content-Type: multipart/form-data
Authorization: Bearer {firebase-id-token}
```

**Request Body** (multipart):
```
file: <binary-image-data>
filter: "ups_logo"  # Opciones: ups_logo, vintage, black_white, sepia
```

**Response** (200 OK):
```json
{
  "previewBase64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "filter": "ups_logo",
  "processedAt": "2025-12-08T20:30:00Z"
}
```

**Nota**: La imagen procesada se retorna en Base64 para mostrar preview en el cliente.

---

### **3. Crear Post**

Crea un nuevo post con metadata en Firestore.

#### **Endpoint**: `POST /api/posts`

**Headers**:
```http
Content-Type: application/json
Authorization: Bearer {firebase-id-token}
```

**Request Body**:
```json
{
  "imageUrl": "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/posts/user123-1733685464.jpg",
  "caption": "Día increíble en la UPS! 🎓✨",
  "filter": "ups_logo"
}
```

**Response** (201 Created):
```json
{
  "id": "post-abc123",
  "userId": "firebase-uid-123",
  "imageUrl": "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/posts/user123-1733685464.jpg",
  "caption": "Día increíble en la UPS! 🎓✨",
  "filter": "ups_logo",
  "likesCount": 0,
  "commentsCount": 0,
  "createdAt": 1733685464000
}
```

---

### **4. Obtener Feed**

Obtiene el feed de posts (últimos 20, ordenados por fecha).

#### **Endpoint**: `GET /api/feed`

**Headers**:
```http
Authorization: Bearer {firebase-id-token}
```

**Query Parameters** (opcionales):
```
limit: 20  (default)
page: 0    (default)
```

**Response** (200 OK):
```json
{
  "posts": [
    {
      "id": "post-abc123",
      "userId": "firebase-uid-123",
      "username": "juanperez",
      "userPhotoUrl": "https://...",
      "imageUrl": "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/posts/...",
      "caption": "Día increíble en la UPS! 🎓✨",
      "filter": "ups_logo",
      "likesCount": 25,
      "commentsCount": 8,
      "isLikedByMe": true,
      "createdAt": 1733685464000
    }
  ],
  "hasMore": true,
  "nextPage": 1
}
```

---

### **5. Dar Like a Post**

#### **Endpoint**: `POST /api/posts/{postId}/likes`

**Headers**:
```http
Authorization: Bearer {firebase-id-token}
```

**Response** (200 OK):
```json
{
  "postId": "post-abc123",
  "userId": "firebase-uid-123",
  "createdAt": 1733685464000
}
```

**Errores**:
- `409 CONFLICT`: Ya diste like a este post

---

### **6. Quitar Like**

#### **Endpoint**: `DELETE /api/posts/{postId}/likes`

**Headers**:
```http
Authorization: Bearer {firebase-id-token}
```

**Response** (204 No Content)

---

### **7. Obtener Likes de un Post**

#### **Endpoint**: `GET /api/posts/{postId}/likes`

**Response** (200 OK):
```json
{
  "likes": [
    {
      "userId": "firebase-uid-123",
      "username": "juanperez",
      "photoUrl": "https://...",
      "createdAt": 1733685464000
    }
  ],
  "total": 25
}
```

---

### **8. Crear Comentario**

#### **Endpoint**: `POST /api/posts/{postId}/comments`

**Headers**:
```http
Content-Type: application/json
Authorization: Bearer {firebase-id-token}
```

**Request Body**:
```json
{
  "text": "Excelente foto! 📸"
}
```

**Response** (201 Created):
```json
{
  "id": "comment-xyz789",
  "postId": "post-abc123",
  "userId": "firebase-uid-123",
  "username": "juanperez",
  "userPhotoUrl": "https://...",
  "text": "Excelente foto! 📸",
  "createdAt": 1733685464000
}
```

---

### **9. Obtener Comentarios de un Post**

#### **Endpoint**: `GET /api/posts/{postId}/comments`

**Response** (200 OK):
```json
{
  "comments": [
    {
      "id": "comment-xyz789",
      "userId": "firebase-uid-123",
      "username": "juanperez",
      "userPhotoUrl": "https://...",
      "text": "Excelente foto! 📸",
      "createdAt": 1733685464000
    }
  ],
  "total": 8
}
```

---

### **10. Eliminar Comentario**

#### **Endpoint**: `DELETE /api/posts/{postId}/comments/{commentId}`

**Headers**:
```http
Authorization: Bearer {firebase-id-token}
```

**Response** (204 No Content)

**Nota**: Solo el autor del comentario o el dueño del post pueden eliminarlo.

---

### **11. Obtener Posts de un Usuario**

#### **Endpoint**: `GET /api/posts/user/{userId}`

**Response** (200 OK):
```json
{
  "posts": [
    {
      "id": "post-abc123",
      "imageUrl": "https://...",
      "caption": "Mi foto favorita",
      "likesCount": 25,
      "commentsCount": 8,
      "createdAt": 1733685464000
    }
  ],
  "total": 42
}
```

---

### **12. Obtener Post por ID**

#### **Endpoint**: `GET /api/posts/{postId}`

**Response** (200 OK):
```json
{
  "id": "post-abc123",
  "userId": "firebase-uid-123",
  "username": "juanperez",
  "userPhotoUrl": "https://...",
  "imageUrl": "https://...",
  "caption": "Día increíble en la UPS! 🎓✨",
  "filter": "ups_logo",
  "likesCount": 25,
  "commentsCount": 8,
  "isLikedByMe": true,
  "createdAt": 1733685464000
}
```

---

### **13. Eliminar Post**

#### **Endpoint**: `DELETE /api/posts/{postId}`

**Headers**:
```http
Authorization: Bearer {firebase-id-token}
```

**Response** (204 No Content)

**Nota**: Solo el autor del post puede eliminarlo.

---

### **14. Actualizar Caption de Post**

#### **Endpoint**: `PATCH /api/posts/{postId}/caption`

**Headers**:
```http
Content-Type: application/json
Authorization: Bearer {firebase-id-token}
```

**Request Body**:
```json
{
  "caption": "Nuevo caption actualizado! 🎉"
}
```

**Response** (200 OK):
```json
{
  "id": "post-abc123",
  "caption": "Nuevo caption actualizado! 🎉",
  "updatedAt": 1733685464000
}
```

---

## 📊 Flujos Completos

### **Flujo 1: Publicar sin Filtro**

```
1. Usuario toma/selecciona foto
2. App → POST /api/images/upload (multipart)
3. Backend → Supabase Storage
4. Backend ← URL pública
5. App → POST /api/posts { imageUrl, caption }
6. Backend → Firestore (metadata)
7. App ← Post creado
8. Feed muestra imagen desde Supabase CDN
```

### **Flujo 2: Publicar con Filtro**

```
1. Usuario toma/selecciona foto
2. Usuario selecciona filtro "ups_logo"
3. App → POST /api/images/preview (multipart + filtro)
4. Backend → PyCUDA Service (GPU)
5. Backend ← Imagen filtrada (Base64)
6. App ← Preview mostrado
7. Usuario confirma "Publicar"
8. App → POST /api/images/upload (imagen filtrada)
9. Backend → Supabase Storage
10. Backend ← URL pública
11. App → POST /api/posts { imageUrl, caption, filter }
12. Backend → Firestore (metadata)
13. App ← Post creado
```

---

## 🐛 Troubleshooting

### **Error: Supabase API key not valid**

**Solución**: Verifica que las variables de entorno estén configuradas correctamente:
```powershell
$env:SUPABASE_URL = "https://opohishcukgkrkfdsgoa.supabase.co"
$env:SUPABASE_KEY = "eyJhbGci..."
```

### **Error: Firebase credentials not found**

**Solución**: Verifica que `firebase-credentials.json` esté en `src/main/resources/`

### **Error: Image upload failed**

**Causas comunes**:
- Archivo muy grande (máximo 10MB configurado)
- Formato no soportado (solo JPG, PNG, WEBP)
- Bucket Supabase no accesible

**Solución**:
```powershell
# Verificar que el bucket 'upsglam' exista en Supabase
# Verificar políticas de Storage en Supabase Console
```

### **Error: PyCUDA Service not available**

**Síntoma**: Preview con filtro falla con timeout.

**Solución**:
1. Verifica que PyCUDA Service esté corriendo en `http://localhost:5000`
2. Si no está disponible, los endpoints de posts siguen funcionando (solo sin filtros)

---

## 📈 Métricas y Logs

### **Health Check**

```bash
GET http://localhost:8081/api/health
```

**Response**:
```json
{
  "status": "UP",
  "services": {
    "firestore": "UP",
    "supabase": "UP",
    "pycuda": "UP"  // Optional
  }
}
```

### **Logs Importantes**

```
INFO  - Subiendo imagen a Supabase: user123-1733685464.jpg
INFO  - Imagen subida exitosamente: https://...
INFO  - Creando post: post-abc123
INFO  - Post creado exitosamente para usuario: firebase-uid-123
INFO  - Like agregado: post-abc123 por firebase-uid-456
INFO  - Comentario creado: comment-xyz789
```

---

## 🎯 Resumen de Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/images/upload` | Subir imagen a Supabase |
| `POST` | `/api/images/preview` | Preview con filtro (PyCUDA) |
| `POST` | `/api/posts` | Crear post |
| `GET` | `/api/feed` | Obtener feed |
| `GET` | `/api/posts/{id}` | Obtener post por ID |
| `GET` | `/api/posts/user/{userId}` | Posts de usuario |
| `DELETE` | `/api/posts/{id}` | Eliminar post |
| `PATCH` | `/api/posts/{id}/caption` | Actualizar caption |
| `POST` | `/api/posts/{id}/likes` | Dar like |
| `DELETE` | `/api/posts/{id}/likes` | Quitar like |
| `GET` | `/api/posts/{id}/likes` | Obtener likes |
| `POST` | `/api/posts/{id}/comments` | Crear comentario |
| `GET` | `/api/posts/{id}/comments` | Obtener comentarios |
| `DELETE` | `/api/posts/{id}/comments/{commentId}` | Eliminar comentario |

---

## 🔐 Autenticación

**Todos los endpoints** (excepto `/health`) requieren autenticación con Firebase ID Token:

```http
Authorization: Bearer {firebase-id-token}
```

El token se obtiene del Auth Service (`POST /api/auth/login`).

---

## 🎓 Notas para el Proyecto UPS

- **Firestore** se usa para metadata porque permite queries en tiempo real
- **Supabase Storage** se usa para imágenes porque es más económico y rápido
- **PyCUDA Service** es opcional - si no está disponible, solo se desactivan los filtros
- Todos los endpoints usan **Spring WebFlux** para operaciones reactivas y no-bloqueantes
- Las imágenes en Supabase se sirven desde CDN público (sin autenticación adicional)
