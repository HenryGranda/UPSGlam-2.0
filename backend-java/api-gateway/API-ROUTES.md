# API Gateway - Routes Configuration

## 🌐 Gateway URL
`http://localhost:8080`

---

## 📋 Arquitectura

```
Mobile App / Frontend
        ↓
API Gateway (Port 8080)
        ↓
┌───────────┬───────────────┬──────────────┐
│ Auth      │ Post Service  │ CUDA Service │
│ Service   │ (Port 8081)   │ (Port 5000)  │
│ (Port     │               │              │
│ 8082)     │               │              │
└───────────┴───────────────┴──────────────┘
```

---

## 🔐 AUTH SERVICE ROUTES

### Base URL: `http://localhost:8080/api/auth`

| Method | Endpoint | Description | Backend Port |
|--------|----------|-------------|--------------|
| POST | `/api/auth/register` | Registrar usuario | 8082 |
| POST | `/api/auth/login` | Iniciar sesión | 8082 |
| GET | `/api/auth/me` | Obtener usuario actual | 8082 |
| GET | `/api/users/{userId}` | Obtener perfil de usuario | 8082 |
| PUT | `/api/users/{userId}` | Actualizar perfil | 8082 |

---

## 📸 POST SERVICE ROUTES

### Base URL: `http://localhost:8080/api`

### 🖼️ Images (Multipart Upload)

| Method | Endpoint | Description | Content-Type | Backend Port |
|--------|----------|-------------|--------------|--------------|
| POST | `/api/images/upload` | Subir imagen a Supabase | `multipart/form-data` | 8081 |
| POST | `/api/images/preview` | Preview con filtro (PyCUDA) | `multipart/form-data` | 8081 |

**Request Example (upload):**
```bash
curl -X POST http://localhost:8080/api/images/upload \
  -H "X-User-Id: user123" \
  -F "image=@/path/to/image.jpg"
```

**Response:**
```json
{
  "imageId": "user123-1733614800000.jpg",
  "imageUrl": "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/posts/user123-1733614800000.jpg"
}
```

---

### 📝 Posts

| Method | Endpoint | Description | Body | Backend Port |
|--------|----------|-------------|------|--------------|
| GET | `/api/feed` | Obtener feed de posts | - | 8081 |
| POST | `/api/posts` | Crear post | JSON | 8081 |
| GET | `/api/posts/{postId}` | Obtener post por ID | - | 8081 |
| DELETE | `/api/posts/{postId}` | Eliminar post | - | 8081 |
| PATCH | `/api/posts/{postId}/caption` | Actualizar descripción | JSON | 8081 |
| GET | `/api/posts/user/{userId}` | Posts de un usuario | - | 8081 |

**Create Post Example:**
```bash
curl -X POST http://localhost:8080/api/posts \
  -H "X-User-Id: user123" \
  -H "X-Username: johndoe" \
  -H "Content-Type: application/json" \
  -d '{
    "imageUrl": "https://supabase.co/.../image.jpg",
    "filter": "ups_logo",
    "caption": "Mi nuevo post!"
  }'
```

---

### ❤️ Likes

| Method | Endpoint | Description | Backend Port |
|--------|----------|-------------|--------------|
| POST | `/api/posts/{postId}/likes` | Dar like | 8081 |
| DELETE | `/api/posts/{postId}/likes` | Quitar like | 8081 |
| GET | `/api/posts/{postId}/likes` | Obtener likes del post | 8081 |

---

### 💬 Comments

| Method | Endpoint | Description | Body | Backend Port |
|--------|----------|-------------|------|--------------|
| POST | `/api/posts/{postId}/comments` | Crear comentario | JSON | 8081 |
| GET | `/api/posts/{postId}/comments` | Obtener comentarios | - | 8081 |
| DELETE | `/api/posts/{postId}/comments/{commentId}` | Eliminar comentario | - | 8081 |
| GET | `/api/users/{userId}/comments` | Comentarios de usuario | - | 8081 |

**Add Comment Example:**
```bash
curl -X POST http://localhost:8080/api/posts/abc123/comments \
  -H "X-User-Id: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Qué bonita foto!",
    "username": "johndoe",
    "userPhotoUrl": "https://..."
  }'
```

---

## 🎨 CUDA SERVICE ROUTES

### Base URL: `http://localhost:8080/api/filters`

| Method | Endpoint | Description | Backend Port |
|--------|----------|-------------|--------------|
| POST | `/api/filters/apply` | Aplicar filtro con GPU | 5000 |
| GET | `/api/filters/list` | Listar filtros disponibles | 5000 |

**Apply Filter Example:**
```bash
curl -X POST http://localhost:8080/api/filters/apply \
  -H "Content-Type: application/json" \
  -d '{
    "imageData": "base64_encoded_image",
    "filter": "ups_logo"
  }'
```

---

## 🏥 HEALTH CHECKS

| Method | Endpoint | Description | Backend |
|--------|----------|-------------|---------|
| GET | `/api/health/auth` | Health check Auth Service | 8082/actuator/health |
| GET | `/api/health/posts` | Health check Post Service | 8081/actuator/health |
| GET | `/api/health/cuda` | Health check CUDA Service | 5000/health |

**Test Health:**
```powershell
# Gateway health
Invoke-RestMethod http://localhost:8080/actuator/health

# Auth service through gateway
Invoke-RestMethod http://localhost:8080/api/health/auth

# Post service through gateway
Invoke-RestMethod http://localhost:8080/api/health/posts

# CUDA service through gateway
Invoke-RestMethod http://localhost:8080/api/health/cuda
```

---

## 🔧 CORS Configuration

**Allowed Origins:** `*` (todas)  
**Allowed Methods:** GET, POST, PUT, DELETE, PATCH, OPTIONS  
**Allowed Headers:** `*` (todos)  
**Exposed Headers:** Content-Type, Authorization  
**Max Age:** 3600 segundos

---

## 🚀 Testing Full Flow

### Scenario: Create Post with Image

```powershell
# Step 1: Upload image to Supabase
$client = New-Object System.Net.Http.HttpClient
$client.DefaultRequestHeaders.Add("X-User-Id", "user123")

$content = New-Object System.Net.Http.MultipartFormDataContent
$fileStream = [System.IO.File]::OpenRead("C:\path\to\image.jpg")
$fileContent = New-Object System.Net.Http.StreamContent($fileStream)
$fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("image/jpeg")
$content.Add($fileContent, "image", "photo.jpg")

$uploadResponse = $client.PostAsync("http://localhost:8080/api/images/upload", $content).Result
$uploadResult = ($uploadResponse.Content.ReadAsStringAsync().Result | ConvertFrom-Json)
$imageUrl = $uploadResult.imageUrl
$fileStream.Close()

Write-Host "Image uploaded: $imageUrl" -ForegroundColor Green

# Step 2: Create post with image URL
$postBody = @{
    imageUrl = $imageUrl
    filter = "none"
    caption = "My new post via API Gateway!"
} | ConvertTo-Json

$postResponse = Invoke-RestMethod `
    -Uri "http://localhost:8080/api/posts" `
    -Method POST `
    -Body $postBody `
    -ContentType "application/json" `
    -Headers @{
        "X-User-Id" = "user123"
        "X-Username" = "testuser"
    }

Write-Host "Post created: $($postResponse.id)" -ForegroundColor Green

# Step 3: Get feed
$feed = Invoke-RestMethod `
    -Uri "http://localhost:8080/api/feed?userId=user123&limit=10" `
    -Method GET `
    -Headers @{"X-User-Id" = "user123"}

Write-Host "Feed retrieved: $($feed.totalElements) posts" -ForegroundColor Cyan

$client.Dispose()
```

---

## 📊 Request Flow Diagram

```
┌─────────────┐
│ Mobile App  │
└──────┬──────┘
       │
       │ POST /api/images/upload
       ↓
┌──────────────────┐
│  API Gateway     │ Port 8080
│  (Routes)        │
└──────┬───────────┘
       │
       │ Forward to http://localhost:8081
       ↓
┌──────────────────┐
│  Post Service    │ Port 8081
│  (Spring Boot)   │
└──────┬───────────┘
       │
       ├─→ Supabase Storage (images)
       │
       └─→ Firestore (metadata)
```

---

## 🐛 Troubleshooting

### Gateway no responde
```bash
# Verificar que el gateway esté corriendo
curl http://localhost:8080/actuator/health
```

### 404 Not Found
- Verificar que el path sea correcto
- Verificar que el servicio backend esté corriendo
- Revisar logs del gateway: `logs/api-gateway.log`

### 503 Service Unavailable
- El servicio backend no está disponible
- Verificar puertos: 8081 (posts), 8082 (auth), 5000 (cuda)
- Probar health check directo del servicio

### CORS Error
- Verificar que `globalcors` esté configurado en `application.yml`
- Verificar headers en la request

---

## 📝 Notes

- **Puerto Gateway:** 8080
- **Puerto Auth Service:** 8082
- **Puerto Post Service:** 8081
- **Puerto CUDA Service:** 5000
- **Todos los requests deben pasar por el Gateway**
- **Headers requeridos:** `X-User-Id` para la mayoría de endpoints
- **Multipart uploads:** Usar `Content-Type: multipart/form-data`
- **JSON requests:** Usar `Content-Type: application/json`

---

**Última actualización:** 7 de diciembre de 2025
