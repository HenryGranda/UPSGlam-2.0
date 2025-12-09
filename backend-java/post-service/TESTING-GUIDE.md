# 🧪 Guía para probar el Post Service localmente (SIN BASE DE DATOS)

**NOTA IMPORTANTE:** Como no hay base de datos configurada, los endpoints responderán con errores de conexión.
Para probarlos de verdad, necesitas:
1. Configurar Supabase PostgreSQL en application.yml
2. O usar una BD PostgreSQL local

## 📍 Servidor corriendo en: http://localhost:8081

---

## 🔍 Probar con PowerShell (Invoke-RestMethod)

### 1. Health Check (debería funcionar)
```powershell
Invoke-RestMethod -Uri "http://localhost:8081/actuator/health" -Method Get
```

### 2. Obtener Feed (FALLARÁ sin BD)
```powershell
$headers = @{
    "X-User-Id" = "test-user-123"
    "Content-Type" = "application/json"
}

Invoke-RestMethod -Uri "http://localhost:8081/feed?page=0&size=10" `
    -Method Get `
    -Headers $headers
```

### 3. Crear Post (FALLARÁ sin BD)
```powershell
$headers = @{
    "X-User-Id" = "test-user-123"
    "X-Username" = "pepito"
    "Content-Type" = "application/json"
}

$body = @{
    mediaUrl = "https://example.com/image.jpg"
    filter = "gaussian"
    caption = "Mi primer post de prueba"
    mediaType = "image"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8081/posts" `
    -Method Post `
    -Headers $headers `
    -Body $body
```

### 4. Dar Like a un Post (FALLARÁ sin BD)
```powershell
$headers = @{
    "X-User-Id" = "test-user-123"
    "X-Username" = "pepito"
}

Invoke-RestMethod -Uri "http://localhost:8081/posts/post-123/likes" `
    -Method Post `
    -Headers $headers
```

### 5. Crear Comentario (FALLARÁ sin BD)
```powershell
$headers = @{
    "X-User-Id" = "test-user-123"
    "X-Username" = "pepito"
    "Content-Type" = "application/json"
}

$body = @{
    text = "Qué buena foto! 😎"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8081/posts/post-123/comments" `
    -Method Post `
    -Headers $headers `
    -Body $body
```

---

## 🔧 Usando cURL (si tienes instalado)

### Health Check
```bash
curl http://localhost:8081/actuator/health
```

### Obtener Feed
```bash
curl -X GET "http://localhost:8081/feed?page=0&size=10" \
  -H "X-User-Id: test-user-123"
```

### Crear Post
```bash
curl -X POST "http://localhost:8081/posts" \
  -H "X-User-Id: test-user-123" \
  -H "X-Username: pepito" \
  -H "Content-Type: application/json" \
  -d '{
    "mediaUrl": "https://example.com/image.jpg",
    "filter": "gaussian",
    "caption": "Mi primer post de prueba",
    "mediaType": "image"
  }'
```

---

## 📋 Lista de todos los endpoints disponibles

### Posts
- `GET /feed?page=0&size=10` - Feed paginado
- `GET /posts/{postId}` - Detalle de post
- `GET /posts/user/{userId}?page=0&size=10` - Posts de usuario
- `POST /posts` - Crear post
- `DELETE /posts/{postId}` - Eliminar post
- `PATCH /posts/{postId}/caption` - Actualizar descripción

### Media
- `POST /images/preview` - Preview con filtro (multipart)
- `POST /images/upload` - Subir imagen (multipart)

### Likes
- `POST /posts/{postId}/likes` - Dar like
- `DELETE /posts/{postId}/likes` - Quitar like
- `GET /posts/{postId}/likes?page=0&size=20` - Lista de likes

### Comments
- `GET /posts/{postId}/comments?page=0&size=20` - Comentarios
- `POST /posts/{postId}/comments` - Crear comentario
- `DELETE /posts/{postId}/comments/{commentId}` - Eliminar comentario
- `GET /users/{userId}/comments?page=0&size=20` - Comentarios de usuario

### Actuator (monitoreo)
- `GET /actuator/health` - Estado de salud
- `GET /actuator/info` - Información del servicio
- `GET /actuator/metrics` - Métricas

---

## ⚙️ Para probar REALMENTE (necesitas configurar BD):

1. **Opción 1: Supabase (según tu arquitectura)**
   - Crea proyecto en Supabase
   - Copia las credenciales
   - Edita `src/main/resources/application.yml`:
   ```yaml
   spring:
     r2dbc:
       url: r2dbc:postgresql://db.xxxxx.supabase.co:5432/postgres
       username: postgres
       password: tu-password
   ```

2. **Opción 2: PostgreSQL local con Docker**
   ```powershell
   docker run -d `
     --name postgres-upsglam `
     -e POSTGRES_PASSWORD=postgres `
     -e POSTGRES_DB=upsglam `
     -p 5432:5432 `
     postgres:15
   ```
   
   Luego edita `application.yml`:
   ```yaml
   spring:
     r2dbc:
       url: r2dbc:postgresql://localhost:5432/upsglam
       username: postgres
       password: postgres
   ```

3. **Ejecuta el SQL para crear las tablas** (revisa schema.sql)

4. **Reinicia la aplicación**

---

## 🛑 Detener el servidor

En la terminal donde está corriendo, presiona `Ctrl + C`

O busca el proceso:
```powershell
Get-Process | Where-Object {$_.ProcessName -like "*java*"} | Stop-Process
```
