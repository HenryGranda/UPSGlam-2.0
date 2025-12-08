# Post Service - API Documentation & Tests

## 🚀 Arquitectura
- **Base de datos**: Firestore (Native Mode) - Database: `db-auth`
- **Almacenamiento**: Supabase Storage - Bucket: `upsglam`
- **Puerto**: 8081
- **Pattern**: Reactive (Mono/Flux)

---

## 📋 Endpoints Disponibles

### 1. **POST /api/posts** - Crear Post
Crea un nuevo post y lo guarda en Firestore.

**Headers:**
```
X-User-Id: user123
X-Username: testuser
Content-Type: application/json
```

**Body:**
```json
{
  "tempImageId": "test123",
  "filter": "ups_logo",
  "caption": "Descripción del post",
  "mediaUrl": "https://example.com/image.jpg",
  "username": "testuser",
  "userPhotoUrl": "https://example.com/avatar.jpg"
}
```

**PowerShell Test:**
```powershell
$headers = @{
    "X-User-Id" = "user123"
    "X-Username" = "testuser"
}
$body = @{
    tempImageId = "test123"
    filter = "ups_logo"
    caption = "Mi primer post con Firestore"
    mediaUrl = "https://example.com/image.jpg"
    username = "testuser"
    userPhotoUrl = "https://example.com/avatar.jpg"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8081/api/posts" `
    -Method POST `
    -Body $body `
    -ContentType "application/json" `
    -Headers $headers
```

**Response:**
```json
{
  "id": "25b92bc5-0566-448f-a3a3-3474b0fee3cb",
  "userId": "user123",
  "username": "testuser",
  "userPhotoUrl": "https://example.com/avatar.jpg",
  "imageUrl": "https://example.com/image.jpg",
  "filter": "ups_logo",
  "description": "Mi primer post con Firestore",
  "createdAt": [2025, 12, 7, 19, 30, 15],
  "likesCount": 0,
  "commentsCount": 0,
  "likedByMe": false
}
```

---

### 2. **GET /api/posts/{postId}** - Obtener Post por ID
Obtiene los detalles de un post específico.

**Headers:**
```
X-User-Id: user123
```

**PowerShell Test:**
```powershell
$postId = "25b92bc5-0566-448f-a3a3-3474b0fee3cb"
$headers = @{"X-User-Id" = "user123"}

Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId" `
    -Headers $headers
```

**Response:**
```json
{
  "id": "25b92bc5-0566-448f-a3a3-3474b0fee3cb",
  "username": "testuser",
  "filter": "ups_logo",
  "description": "Mi primer post con Firestore",
  "likesCount": 0,
  "commentsCount": 0,
  "likedByMe": false
}
```

---

### 3. **GET /api/feed** - Obtener Feed Paginado
Obtiene el feed de posts ordenados por fecha de creación (más recientes primero).

**Query Parameters:**
- `page`: Número de página (default: 0)
- `size`: Tamaño de página (default: 10)

**Headers:**
```
X-User-Id: user123
```

**PowerShell Test:**
```powershell
$headers = @{"X-User-Id" = "user123"}

Invoke-RestMethod -Uri "http://localhost:8081/api/feed?page=0&size=10" `
    -Headers $headers
```

**Response:**
```json
{
  "posts": [
    {
      "id": "post-id-1",
      "username": "maria",
      "description": "Post más reciente",
      "likesCount": 5,
      "commentsCount": 2,
      "likedByMe": true
    }
  ],
  "page": 0,
  "size": 10,
  "totalItems": 15,
  "hasMore": true
}
```

---

### 4. **DELETE /api/posts/{postId}** - Eliminar Post
Elimina un post (solo el autor puede eliminarlo).

**Headers:**
```
X-User-Id: user123
```

**PowerShell Test:**
```powershell
$postId = "25b92bc5-0566-448f-a3a3-3474b0fee3cb"
$headers = @{"X-User-Id" = "user123"}

Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId" `
    -Method DELETE `
    -Headers $headers

# Response: 204 No Content
```

---

## ❤️ Likes

### 5. **POST /api/posts/{postId}/likes** - Dar Like
Agrega un like al post y crea documento en `posts/{postId}/likes/{userId}`.

**Headers:**
```
X-User-Id: user123
```

**PowerShell Test:**
```powershell
$postId = "25b92bc5-0566-448f-a3a3-3474b0fee3cb"
$headers = @{"X-User-Id" = "user123"}

Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId/likes" `
    -Method POST `
    -Headers $headers
```

**Response:**
```json
{
  "postId": "25b92bc5-0566-448f-a3a3-3474b0fee3cb",
  "userId": "user123",
  "liked": true,
  "likesCount": 1,
  "createdAt": [2025, 12, 7, 19, 35, 20]
}
```

---

### 6. **DELETE /api/posts/{postId}/likes** - Quitar Like
Elimina el like del post.

**Headers:**
```
X-User-Id: user123
```

**PowerShell Test:**
```powershell
$postId = "25b92bc5-0566-448f-a3a3-3474b0fee3cb"
$headers = @{"X-User-Id" = "user123"}

Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId/likes" `
    -Method DELETE `
    -Headers $headers
```

**Response:**
```json
{
  "postId": "25b92bc5-0566-448f-a3a3-3474b0fee3cb",
  "userId": "user123",
  "liked": false,
  "likesCount": 0,
  "createdAt": [2025, 12, 7, 19, 36, 10]
}
```

---

## 💬 Comments

### 7. **POST /api/posts/{postId}/comments** - Agregar Comentario
Agrega un comentario al post en `posts/{postId}/comments/{commentId}`.

**Headers:**
```
X-User-Id: user123
Content-Type: application/json
```

**Body:**
```json
{
  "text": "¡Excelente post! 🎉",
  "username": "testuser",
  "userPhotoUrl": "https://example.com/avatar.jpg"
}
```

**PowerShell Test:**
```powershell
$postId = "25b92bc5-0566-448f-a3a3-3474b0fee3cb"
$headers = @{"X-User-Id" = "user123"}
$body = @{
    text = "¡Excelente post! Firestore funcionando perfectamente 🎉"
    username = "testuser"
    userPhotoUrl = "https://example.com/avatar.jpg"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId/comments" `
    -Method POST `
    -Body $body `
    -ContentType "application/json" `
    -Headers $headers
```

**Response:**
```json
{
  "id": "7a063cc0-d106-4eda-8e50-4a2e9a2a6884",
  "postId": "25b92bc5-0566-448f-a3a3-3474b0fee3cb",
  "userId": "user123",
  "username": "testuser",
  "userPhotoUrl": "https://example.com/avatar.jpg",
  "text": "¡Excelente post! Firestore funcionando perfectamente 🎉",
  "createdAt": [2025, 12, 7, 19, 37, 30]
}
```

---

### 8. **GET /api/posts/{postId}/comments** - Obtener Comentarios
Obtiene todos los comentarios de un post (paginado).

**Query Parameters:**
- `page`: Número de página (default: 0)
- `size`: Tamaño de página (default: 50)

**Headers:**
```
X-User-Id: user123
```

**PowerShell Test:**
```powershell
$postId = "25b92bc5-0566-448f-a3a3-3474b0fee3cb"
$headers = @{"X-User-Id" = "user123"}

Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId/comments" `
    -Headers $headers
```

**Response:**
```json
{
  "postId": "25b92bc5-0566-448f-a3a3-3474b0fee3cb",
  "comments": [
    {
      "id": "7a063cc0-d106-4eda-8e50-4a2e9a2a6884",
      "userId": "user123",
      "username": "testuser",
      "text": "¡Excelente post!",
      "createdAt": [2025, 12, 7, 19, 37, 30]
    }
  ],
  "totalCount": 1
}
```

---

### 9. **DELETE /api/posts/{postId}/comments/{commentId}** - Eliminar Comentario
Elimina un comentario (solo el autor puede eliminarlo).

**Headers:**
```
X-User-Id: user123
```

**PowerShell Test:**
```powershell
$postId = "25b92bc5-0566-448f-a3a3-3474b0fee3cb"
$commentId = "7a063cc0-d106-4eda-8e50-4a2e9a2a6884"
$headers = @{"X-User-Id" = "user123"}

Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId/comments/$commentId" `
    -Method DELETE `
    -Headers $headers

# Response: 204 No Content
```

---

## 🧪 Test Completo (Suite de Pruebas)

```powershell
# ====================================
# SUITE DE PRUEBAS COMPLETA
# ====================================

Write-Host "`n=== 1. CREAR POST ===" -ForegroundColor Cyan
$headers = @{"X-User-Id"="user123"; "X-Username"="testuser"}
$body = '{"tempImageId":"test123","filter":"ups_logo","caption":"Post de prueba completo","mediaUrl":"https://example.com/image.jpg","username":"testuser","userPhotoUrl":"https://example.com/avatar.jpg"}'
$post = Invoke-RestMethod -Uri "http://localhost:8081/api/posts" -Method POST -Body $body -ContentType "application/json" -Headers $headers
$postId = $post.id
Write-Host "✓ Post creado: $postId" -ForegroundColor Green

Write-Host "`n=== 2. OBTENER POST ===" -ForegroundColor Cyan
$post = Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId" -Headers @{"X-User-Id"="user123"}
Write-Host "✓ Post obtenido: $($post.description)" -ForegroundColor Green

Write-Host "`n=== 3. DAR LIKE ===" -ForegroundColor Cyan
$like = Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId/likes" -Method POST -Headers @{"X-User-Id"="user123"}
Write-Host "✓ Like agregado. Total: $($like.likesCount)" -ForegroundColor Green

Write-Host "`n=== 4. VERIFICAR LIKED BY ME ===" -ForegroundColor Cyan
$post = Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId" -Headers @{"X-User-Id"="user123"}
Write-Host "✓ Liked by me: $($post.likedByMe)" -ForegroundColor Green

Write-Host "`n=== 5. AGREGAR COMENTARIO ===" -ForegroundColor Cyan
$commentBody = '{"text":"Comentario de prueba","username":"testuser","userPhotoUrl":"https://example.com/avatar.jpg"}'
$comment = Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId/comments" -Method POST -Body $commentBody -ContentType "application/json" -Headers @{"X-User-Id"="user123"}
$commentId = $comment.id
Write-Host "✓ Comentario agregado: $commentId" -ForegroundColor Green

Write-Host "`n=== 6. OBTENER COMENTARIOS ===" -ForegroundColor Cyan
$comments = Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId/comments" -Headers @{"X-User-Id"="user123"}
Write-Host "✓ Total comentarios: $($comments.totalCount)" -ForegroundColor Green

Write-Host "`n=== 7. OBTENER FEED ===" -ForegroundColor Cyan
$feed = Invoke-RestMethod -Uri "http://localhost:8081/api/feed?page=0&size=10" -Headers @{"X-User-Id"="user123"}
Write-Host "✓ Posts en feed: $($feed.totalItems)" -ForegroundColor Green

Write-Host "`n=== 8. QUITAR LIKE ===" -ForegroundColor Cyan
$unlike = Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId/likes" -Method DELETE -Headers @{"X-User-Id"="user123"}
Write-Host "✓ Like removido. Total: $($unlike.likesCount)" -ForegroundColor Green

Write-Host "`n=== 9. ELIMINAR COMENTARIO ===" -ForegroundColor Cyan
Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId/comments/$commentId" -Method DELETE -Headers @{"X-User-Id"="user123"}
Write-Host "✓ Comentario eliminado" -ForegroundColor Green

Write-Host "`n=== 10. ELIMINAR POST ===" -ForegroundColor Cyan
Invoke-RestMethod -Uri "http://localhost:8081/api/posts/$postId" -Method DELETE -Headers @{"X-User-Id"="user123"}
Write-Host "✓ Post eliminado" -ForegroundColor Green

Write-Host "`n=== ✅ TODAS LAS PRUEBAS COMPLETADAS ===" -ForegroundColor Green
```

---

## 🔥 Estructura de Firestore

```
db-auth/
├── posts/
│   ├── {postId}/
│   │   ├── id: string
│   │   ├── userId: string
│   │   ├── username: string
│   │   ├── userPhotoUrl: string
│   │   ├── imageUrl: string (URL de Supabase)
│   │   ├── filter: string
│   │   ├── description: string
│   │   ├── createdAt: Timestamp
│   │   ├── likesCount: number
│   │   ├── commentsCount: number
│   │   │
│   │   ├── likes/
│   │   │   └── {userId}/
│   │   │       ├── userId: string
│   │   │       └── createdAt: Timestamp
│   │   │
│   │   └── comments/
│   │       └── {commentId}/
│   │           ├── id: string
│   │           ├── userId: string
│   │           ├── username: string
│   │           ├── userPhotoUrl: string
│   │           ├── text: string
│   │           └── createdAt: Timestamp
```

---

## ⚙️ Configuración

### application-local.yml
```yaml
firebase:
  api-key: AIzaSyBYcnFxABxm3eyFpCD-nioQbZV1-NDzA5A

supabase:
  url: https://ihklfvzdlpxmycxrvjmf.supabase.co
  key: eyJhbGci... (anon key)
  service-role-key: eyJhbGci... (service role key)
```

### Iniciar Servicio
```bash
cd C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\post-service
mvn clean package -DskipTests
java -jar target/post-service-1.0.0.jar --spring.profiles.active=local
```

---

## 📊 Health Check

```powershell
Invoke-RestMethod -Uri "http://localhost:8081/actuator/health"
```

**Response:**
```json
{
  "status": "UP"
}
```

---

## 🎯 Endpoints Implementados vs No Implementados

### ✅ Implementados y Funcionando
- POST /api/posts
- GET /api/posts/{postId}
- GET /api/feed
- DELETE /api/posts/{postId}
- POST /api/posts/{postId}/likes
- DELETE /api/posts/{postId}/likes
- POST /api/posts/{postId}/comments
- GET /api/posts/{postId}/comments
- DELETE /api/posts/{postId}/comments/{commentId}

### ⚠️ Parcialmente Implementados
- GET /api/posts/user/{userId} - Retorna lista vacía
- PATCH /api/posts/{postId}/caption - Retorna 501 Not Implemented
- GET /api/posts/{postId}/likes - Retorna lista vacía
- GET /api/users/{userId}/comments - Retorna lista vacía

### 🔜 Pendientes de Implementar
- POST /api/images/upload - Subir imagen a Supabase Storage
- POST /api/images/preview - Preview temporal de imagen

---

## 🐛 Errores Comunes

### 1. "Failed to obtain R2DBC Connection"
**Causa:** El servicio intenta conectarse a PostgreSQL (viejo).
**Solución:** Asegúrate de usar los repositorios Firestore, no los R2DBC.

### 2. "class java.util.HashMap cannot be cast to com.google.cloud.Timestamp"
**Causa:** Error en conversión de timestamps de Firestore.
**Solución:** Ya corregido con el método `convertToInstant()`.

### 3. "PostNotFoundException"
**Causa:** El post no existe en Firestore.
**Solución:** Verifica que el postId sea correcto.

### 4. "UnauthorizedException"
**Causa:** Intentando eliminar post/comentario de otro usuario.
**Solución:** Usa el mismo X-User-Id del autor.

---

## 📝 Notas

- Todos los IDs se generan automáticamente con UUID
- Los timestamps se almacenan como `Instant` en Java y `Timestamp` en Firestore
- Los contadores (`likesCount`, `commentsCount`) usan `FieldValue.increment()` para operaciones atómicas
- Las subcollections de likes y comments están dentro de cada post
- El campo `likedByMe` se calcula en tiempo real consultando la subcollection de likes

---

**Última actualización:** 7 de diciembre de 2025
