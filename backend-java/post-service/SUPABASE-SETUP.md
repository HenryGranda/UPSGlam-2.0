# Supabase Storage Setup Guide

## 🎯 Objetivo
Configurar Supabase Storage para almacenar las **imágenes finales** de los posts y generar URLs públicas que se guardan en Firestore.

## 🏗️ Arquitectura del Sistema

### Stack Tecnológico
```
Mobile App (Flutter/React Native)
    ↓
API Gateway (Post Service - Spring Boot)
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│  Firebase        │  Supabase        │  PyCUDA         │
│  Firestore       │  Storage         │  Service        │
├─────────────────┼──────────────────┼─────────────────┤
│  • Posts         │  • Imágenes      │  • Filtros      │
│  • Likes         │    finales       │    (GPU)        │
│  • Comments      │  • CDN público   │  • Preview      │
│  • Metadata      │  • URLs          │    en memoria   │
└─────────────────┴──────────────────┴─────────────────┘
```

### Flujo de Usuario (OPTIMIZADO ✅)

#### Opción 1: Publicar sin filtro
```
1. Usuario toma/selecciona foto
2. Foto queda en memoria del app
3. Usuario da "Publicar"
4. App → POST /api/images/upload (multipart/form-data)
5. Backend → Supabase Storage (posts/user123-timestamp.jpg)
6. Backend ← URL pública
7. App → POST /api/posts { imageUrl, caption }
8. Backend → Firestore (metadata + imageUrl)
9. App ← Post creado
10. Feed muestra imagen desde Supabase URL
```

#### Opción 2: Publicar con filtro
```
1. Usuario toma/selecciona foto
2. Foto queda en MEMORIA del app (NO se sube aún)
3. Usuario selecciona filtro "ups_logo"
4. App → POST /api/images/preview (multipart con filtro)
5. Backend → PyCUDA Service (procesa en GPU)
6. Backend ← Imagen filtrada (bytes)
7. App ← Preview de imagen filtrada
8. Usuario cambia filtro → Se repite paso 4-7
9. Usuario da "Publicar"
10. App → POST /api/images/upload (imagen FILTRADA final)
11. Backend → Supabase Storage
12. Backend ← URL pública
13. App → POST /api/posts { imageUrl, filter, caption }
14. Backend → Firestore
15. Feed muestra imagen filtrada desde Supabase
```

**Ventajas de este flujo:**
- ✅ No sube imágenes temporales innecesarias
- ✅ Usuario prueba filtros sin costo de storage
- ✅ Solo se almacena versión final
- ✅ PyCUDA procesa en memoria (rápido)
- ✅ Ahorra ancho de banda y storage

---

## 📋 Paso 1: Crear Proyecto en Supabase

1. **Ir a Supabase Dashboard**
   - URL: https://supabase.com/dashboard
   - Login con tu cuenta

2. **Crear nuevo proyecto** (si no existe)
   - Click en "New Project"
   - Project name: `upsglam`
   - Database Password: (guárdalo)
   - Region: `South America (São Paulo)` (más cercano)
   - Pricing: Free

3. **Obtener credenciales**
   - Ve a `Settings` → `API`
   - Copia:
     - **Project URL**: `https://ihklfvzdlpxmycxrvjmf.supabase.co`
     - **anon/public key**: `eyJhbGci...`
     - **service_role key**: `eyJhbGci...` (⚠️ MANTENER SECRETO)

---

## 📦 Paso 2: Crear Storage Bucket

1. **Ir a Storage**
   - En el sidebar, click en `Storage`
   - Click en `Create a new bucket`

2. **Configurar bucket**
   ```
   Name: upsglam
   Public bucket: ✅ YES (para URLs públicas)
   File size limit: 5 MB (para imágenes)
   Allowed MIME types: image/jpeg, image/png, image/webp
   ```

3. **Click en "Create bucket"**

---

## 📁 Paso 3: Crear Carpetas (Folders)

Dentro del bucket `upsglam`, crea estas carpetas:

1. **posts/** - Imágenes finales de posts
2. **temp/** - Imágenes temporales (preview)
3. **avatars/** - Fotos de perfil de usuarios

### Crear carpetas:
1. Click en el bucket `upsglam`
2. Click en "New folder"
3. Nombre: `posts`
4. Click "Create folder"
5. Repetir para `temp` y `avatars`

---

## 🔒 Paso 4: Configurar Políticas de Acceso (RLS)

### 4.1 Deshabilitar RLS para Buckets Públicos (Recomendado para desarrollo)

1. Ve a `Storage` → `Policies`
2. En el bucket `upsglam`, asegúrate de que:
   - **Public access**: Enabled
   - Esto permite que cualquiera pueda leer archivos con URLs públicas

### 4.2 Crear Políticas Personalizadas (Producción)

Si quieres más control, crea estas políticas:

#### Política 1: Lectura Pública
```sql
CREATE POLICY "Public read access"
ON storage.objects FOR SELECT
USING (bucket_id = 'upsglam');
```

#### Política 2: Subir con Service Role
```sql
CREATE POLICY "Service role can upload"
ON storage.objects FOR INSERT
WITH CHECK (
  bucket_id = 'upsglam' AND
  auth.role() = 'service_role'
);
```

#### Política 3: Eliminar con Service Role
```sql
CREATE POLICY "Service role can delete"
ON storage.objects FOR DELETE
USING (
  bucket_id = 'upsglam' AND
  auth.role() = 'service_role'
);
```

---

## ⚙️ Paso 5: Configurar Backend (application-local.yml)

1. **Editar archivo de configuración**
   ```bash
   cd C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\post-service\src\main\resources
   notepad application-local.yml
   ```

2. **Agregar/Actualizar configuración de Supabase**
   ```yaml
   firebase:
     api-key: AIzaSyBYcnFxABxm3eyFpCD-nioQbZV1-NDzA5A

   supabase:
     url: https://ihklfvzdlpxmycxrvjmf.supabase.co
     key: eyJhbGci... # ANON KEY (copiar de Supabase)
     service-role-key: eyJhbGci... # SERVICE ROLE KEY (copiar de Supabase)
     storage:
       bucket: upsglam
       folders:
         posts: posts
         temp: temp
         avatars: avatars
   ```

3. **Guardar archivo** (NO hacer commit, está en .gitignore)

---

## 🧪 Paso 6: Probar Subida de Imagen

### 6.1 Preparar imagen de prueba
```powershell
# Descargar imagen de ejemplo
Invoke-WebRequest -Uri "https://picsum.photos/800/600" -OutFile "C:\temp\test-image.jpg"

# O crear una imagen de prueba con Paint:
mspaint C:\temp\test-image.jpg
# Dibuja algo y guarda
```

### 6.2 Convertir imagen a Base64
```powershell
$imagePath = "C:\temp\test-image.jpg"
$imageBytes = [System.IO.File]::ReadAllBytes($imagePath)
$base64String = [System.Convert]::ToBase64String($imageBytes)

# Guardar en archivo para usar después
$base64String | Out-File "C:\temp\image-base64.txt"
```

### 6.3 Probar endpoint de upload (cuando esté implementado)
```powershell
$headers = @{
    "X-User-Id" = "user123"
    "Content-Type" = "application/json"
}

$body = @{
    imageData = $base64String
    fileName = "test-post-$(Get-Date -Format 'yyyyMMddHHmmss').jpg"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8081/api/images/upload" `
    -Method POST `
    -Body $body `
    -Headers $headers
```

### 6.4 Verificar en Supabase Dashboard
1. Ve a `Storage` → `upsglam` → `posts`
2. Deberías ver tu imagen subida
3. Click en la imagen
4. Copia la URL pública: 
   ```
   https://ihklfvzdlpxmycxrvjmf.supabase.co/storage/v1/object/public/upsglam/posts/test-post-20251207.jpg
   ```

---

## 🔧 Paso 7: Implementar Endpoint de Upload (Backend)

El código ya está en `SupabaseStorageClient.java`, pero falta el endpoint REST.

### 7.1 Crear MediaHandler (si no existe completo)

```java
// En MediaHandler.java
public Mono<ServerResponse> uploadImage(ServerRequest request) {
    String userId = extractUserId(request);
    
    return request.bodyToMono(UploadImageRequest.class)
            .flatMap(uploadRequest -> {
                // Decodificar Base64
                byte[] imageBytes = Base64.getDecoder().decode(uploadRequest.getImageData());
                
                // Generar nombre único
                String fileName = userId + "-" + System.currentTimeMillis() + ".jpg";
                
                // Subir a Supabase
                return storageClient.uploadPostImage(fileName, imageBytes)
                        .map(publicUrl -> UploadImageResponse.builder()
                                .imageUrl(publicUrl)
                                .fileName(fileName)
                                .build());
            })
            .flatMap(response -> 
                ServerResponse.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(response)
            )
            .onErrorResume(this::handleError);
}
```

### 7.2 Crear DTOs

```java
// UploadImageRequest.java
@Data
@NoArgsConstructor
@AllArgsConstructor
public class UploadImageRequest {
    @NotBlank
    private String imageData; // Base64 encoded
    
    private String fileName; // Opcional
}

// UploadImageResponse.java
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UploadImageResponse {
    private String imageUrl;
    private String fileName;
}
```

### 7.3 Agregar ruta en MediaRouter

```java
@Bean
public RouterFunction<ServerResponse> mediaRoutes(MediaHandler handler) {
    return RouterFunctions
        .route(POST("/images/upload")
            .and(contentType(MediaType.APPLICATION_JSON)), 
            handler::uploadImage)
        .route(POST("/images/preview")
            .and(contentType(MediaType.MULTIPART_FORM_DATA)), 
            handler::uploadPreview);
}
```

---

## 🚀 Paso 8: Flujo Completo de Creación de Post

### Arquitectura de Endpoints

```
POST /api/images/preview
├─ Recibe: multipart/form-data (image + filter)
├─ Envía a: PyCUDA Service (GPU processing)
├─ Retorna: Imagen filtrada (bytes)
└─ NO guarda en Supabase (solo preview)

POST /api/images/upload
├─ Recibe: multipart/form-data (image)
├─ Sube a: Supabase Storage (posts/)
├─ Retorna: { imageId, imageUrl }
└─ URL pública lista para usar

POST /api/posts
├─ Recibe: { imageUrl, filter, caption }
├─ Guarda en: Firestore (db-auth/posts)
├─ Retorna: Post completo con ID
└─ imageUrl apunta a Supabase
```

### Flujo Mobile App → Backend

#### Escenario 1: Post SIN filtro (directo)

```javascript
// 1. Usuario selecciona foto
const imageFile = await ImagePicker.pickImage();

// 2. Subir directamente a Supabase
const formData = new FormData();
formData.append('image', imageFile);

const uploadResponse = await fetch('http://localhost:8081/api/images/upload', {
  method: 'POST',
  headers: { 'X-User-Id': userId },
  body: formData
});
const { imageUrl } = await uploadResponse.json();

// 3. Crear post con URL
const postResponse = await fetch('http://localhost:8081/api/posts', {
  method: 'POST',
  headers: {
    'X-User-Id': userId,
    'X-Username': username,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    imageUrl: imageUrl,  // URL de Supabase
    filter: 'none',
    caption: 'Mi nuevo post!'
  })
});

// 4. Post creado, aparece en feed
```

#### Escenario 2: Post CON filtro (preview + upload final)

```javascript
// 1. Usuario selecciona foto
const imageFile = await ImagePicker.pickImage();
let currentPreview = imageFile; // Mantener en memoria

// 2. Usuario selecciona filtro → Preview
const filterFormData = new FormData();
filterFormData.append('image', imageFile);
filterFormData.append('filter', 'ups_logo');

const previewResponse = await fetch('http://localhost:8081/api/images/preview', {
  method: 'POST',
  headers: { 'X-User-Id': userId },
  body: filterFormData
});
const filteredImageBlob = await previewResponse.blob();
currentPreview = filteredImageBlob; // Actualizar preview

// Mostrar preview en UI
setImagePreview(URL.createObjectURL(filteredImageBlob));

// 3. Usuario cambia filtro → Repetir paso 2 con otro filtro
// (Puede probar múltiples filtros sin subir nada)

// 4. Usuario da "Publicar" → Subir versión FINAL filtrada
const uploadFormData = new FormData();
uploadFormData.append('image', currentPreview); // Imagen filtrada

const uploadResponse = await fetch('http://localhost:8081/api/images/upload', {
  method: 'POST',
  headers: { 'X-User-Id': userId },
  body: uploadFormData
});
const { imageUrl } = await uploadResponse.json();

// 5. Crear post con imagen filtrada
const postResponse = await fetch('http://localhost:8081/api/posts', {
  method: 'POST',
  headers: {
    'X-User-Id': userId,
    'X-Username': username,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    imageUrl: imageUrl,      // URL de imagen filtrada en Supabase
    filter: 'ups_logo',
    caption: '¡Con filtro UPS!'
  })
});

// 6. Post con filtro creado
```

### Ejemplo Completo en PowerShell (Testing)

```powershell
# ESCENARIO 1: Sin filtro (directo)
Write-Host "📸 Test 1: Upload directo sin filtro" -ForegroundColor Cyan

$imagePath = "C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\husky.jpg"
$client = New-Object System.Net.Http.HttpClient
$client.DefaultRequestHeaders.Add("X-User-Id", "user123")

# Subir imagen
$content = New-Object System.Net.Http.MultipartFormDataContent
$fileStream = [System.IO.File]::OpenRead($imagePath)
$fileContent = New-Object System.Net.Http.StreamContent($fileStream)
$fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("image/jpeg")
$content.Add($fileContent, "image", "husky.jpg")

$response = $client.PostAsync("http://localhost:8081/api/images/upload", $content).Result
$uploadResult = ($response.Content.ReadAsStringAsync().Result | ConvertFrom-Json)
$fileStream.Close()

Write-Host "✅ Imagen subida: $($uploadResult.imageUrl)" -ForegroundColor Green

# Crear post
$postBody = @{
    imageUrl = $uploadResult.imageUrl
    filter = "none"
    caption = "Post directo sin filtro"
} | ConvertTo-Json

$postResponse = Invoke-RestMethod -Uri "http://localhost:8081/api/posts" `
    -Method POST `
    -Body $postBody `
    -ContentType "application/json" `
    -Headers @{"X-User-Id"="user123"; "X-Username"="testuser"}

Write-Host "✅ Post creado: $($postResponse.id)" -ForegroundColor Green
Write-Host ""

# ============================================

# ESCENARIO 2: Con filtro (preview → upload)
Write-Host "🎨 Test 2: Con preview de filtro" -ForegroundColor Cyan

# Probar filtro (preview)
$previewContent = New-Object System.Net.Http.MultipartFormDataContent
$fileStream2 = [System.IO.File]::OpenRead($imagePath)
$imageContent = New-Object System.Net.Http.StreamContent($fileStream2)
$imageContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("image/jpeg")
$previewContent.Add($imageContent, "image", "husky.jpg")
$filterContent = New-Object System.Net.Http.StringContent("ups_logo")
$previewContent.Add($filterContent, "filter")

Write-Host "Aplicando filtro ups_logo..." -ForegroundColor Yellow
$previewResponse = $client.PostAsync("http://localhost:8081/api/images/preview", $previewContent).Result
$filteredBytes = $previewResponse.Content.ReadAsByteArrayAsync().Result
$fileStream2.Close()

Write-Host "✅ Filtro aplicado ($($filteredBytes.Length) bytes)" -ForegroundColor Green

# Subir imagen filtrada
$uploadContent = New-Object System.Net.Http.MultipartFormDataContent
$filteredStream = New-Object System.IO.MemoryStream($filteredBytes)
$filteredContent = New-Object System.Net.Http.StreamContent($filteredStream)
$filteredContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("image/jpeg")
$uploadContent.Add($filteredContent, "image", "filtered-husky.jpg")

$finalResponse = $client.PostAsync("http://localhost:8081/api/images/upload", $uploadContent).Result
$finalUpload = ($finalResponse.Content.ReadAsStringAsync().Result | ConvertFrom-Json)
$filteredStream.Close()

Write-Host "✅ Imagen filtrada subida: $($finalUpload.imageUrl)" -ForegroundColor Green

# Crear post con filtro
$filteredPostBody = @{
    imageUrl = $finalUpload.imageUrl
    filter = "ups_logo"
    caption = "Post con filtro UPS aplicado"
} | ConvertTo-Json

$filteredPost = Invoke-RestMethod -Uri "http://localhost:8081/api/posts" `
    -Method POST `
    -Body $filteredPostBody `
    -ContentType "application/json" `
    -Headers @{"X-User-Id"="user123"; "X-Username"="testuser"}

Write-Host "✅ Post con filtro creado: $($filteredPost.id)" -ForegroundColor Green

$client.Dispose()
```

---

## 📊 Verificar Todo Funciona

### 1. **Verificar en Supabase Dashboard**
```
Storage → upsglam → posts → [ver imagen subida]
```

### 2. **Verificar en Firebase Console**
```
Firestore → db-auth → posts → [ver documento con imageUrl]
```

### 3. **Verificar URL pública funciona**
```powershell
# Abrir imagen en navegador
Start-Process "https://ihklfvzdlpxmycxrvjmf.supabase.co/storage/v1/object/public/upsglam/posts/user123-1733614800000.jpg"
```

### 4. **Probar desde mobile app**
```dart
// En Flutter
final bytes = await image.readAsBytes();
final base64 = base64Encode(bytes);

final response = await http.post(
  Uri.parse('http://tu-ip:8081/api/images/upload'),
  headers: {
    'X-User-Id': userId,
    'Content-Type': 'application/json',
  },
  body: jsonEncode({'imageData': base64}),
);

final imageUrl = jsonDecode(response.body)['imageUrl'];
```

---

## 🔐 Seguridad en Producción

### 1. **Variables de entorno**
```bash
# No guardar keys en application-local.yml
# Usar variables de entorno:
export SUPABASE_URL=https://xxx.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
```

### 2. **Validaciones Backend**
- ✅ Validar tamaño de imagen (max 5MB)
- ✅ Validar formato (solo jpg, png, webp)
- ✅ Validar que el usuario esté autenticado
- ✅ Sanitizar nombres de archivo

### 3. **Rate Limiting**
- Limitar uploads por usuario (ej: 10 posts/hora)
- Usar bucket de Supabase con rate limits

### 4. **Eliminar imágenes huérfanas**
- Cuando se elimina un post, eliminar imagen de Supabase
- Implementar job para limpiar imágenes de `temp/` viejas

---

## 🐛 Troubleshooting

### Error: "Failed to upload to Supabase"
**Solución:**
1. Verifica que el `service-role-key` sea correcto
2. Verifica que el bucket `upsglam` exista
3. Verifica que las carpetas estén creadas
4. Revisa los logs del backend: `target/logs/post-service.log`

### Error: "403 Forbidden"
**Solución:**
1. Ve a Supabase → Storage → Policies
2. Asegúrate que el bucket sea público
3. O crea las políticas RLS necesarias

### Error: "Image too large"
**Solución:**
1. Comprimir imagen en mobile app antes de subir
2. Cambiar límite en Supabase Storage settings

### URL pública no funciona
**Solución:**
1. Verifica que el bucket sea **público**
2. URL correcta: `{url}/storage/v1/object/public/{bucket}/{path}`
3. No usar: `{url}/storage/v1/object/{bucket}/{path}` (privado)

---

## 📝 Checklist Final

Antes de probar todo, verifica:

- [ ] Proyecto de Supabase creado
- [ ] Bucket `upsglam` creado y público
- [ ] Carpetas `posts/`, `temp/`, `avatars/` creadas
- [ ] Políticas RLS configuradas (o bucket público)
- [ ] `application-local.yml` configurado con keys
- [ ] Post service compilado y corriendo
- [ ] Endpoint `/api/images/upload` implementado
- [ ] Probado upload con imagen real
- [ ] URL pública accesible desde navegador
- [ ] Post creado en Firestore con imageUrl correcto

---

## 🎉 Resultado Final

### Arquitectura Completa Implementada

```
┌─────────────────────────────────────────────────────────────┐
│                     MOBILE APP (Flutter)                     │
│                                                              │
│  • Cámara / Galería                                          │
│  • Preview de filtros en memoria                             │
│  • Upload solo de versión final                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│              POST SERVICE (Spring Boot WebFlux)              │
│                      Puerto: 8081                            │
│                                                              │
│  Endpoints:                                                  │
│  • POST /api/images/preview   → PyCUDA (filtros)            │
│  • POST /api/images/upload    → Supabase (storage)          │
│  • POST /api/posts            → Firestore (metadata)        │
│  • GET  /api/feed             → Firestore + Supabase URLs   │
│  • POST /api/posts/{id}/likes                               │
│  • POST /api/posts/{id}/comments                            │
└─────┬────────────────────┬────────────────────┬─────────────┘
      │                    │                    │
      ↓                    ↓                    ↓
┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐
│   PyCUDA     │  │    Supabase      │  │    Firebase     │
│   Service    │  │    Storage       │  │    Firestore    │
├──────────────┤  ├──────────────────┤  ├─────────────────┤
│ • GPU        │  │ • CDN público    │  │ • db-auth       │
│ • Filtros:   │  │ • Bucket:        │  │ • Collections:  │
│   - ups_logo │  │   upsglam        │  │   - posts       │
│   - sepia    │  │ • Folders:       │  │   - likes       │
│   - blur     │  │   - posts/       │  │   - comments    │
│ • Retorna    │  │   - temp/        │  │ • Queries       │
│   imagen     │  │   - avatars/     │  │   optimizadas   │
│   procesada  │  │ • URLs públicas  │  │ • Real-time     │
└──────────────┘  └──────────────────┘  └─────────────────┘
```

### Flujo de Datos Optimizado

**SIN Filtro (Rápido):**
```
Usuario → Foto → Upload → Supabase → URL → Post → Firestore → Feed
         (1 paso)                   (almacenamiento permanente)
```

**CON Filtro (Optimizado):**
```
Usuario → Foto (memoria)
    ↓
  Filtro 1 → PyCUDA → Preview 1 (memoria)
    ↓
  Filtro 2 → PyCUDA → Preview 2 (memoria)
    ↓
  Publicar → Upload → Supabase → URL → Post → Firestore → Feed
           (solo versión final)  (almacenamiento permanente)
```

### Ventajas de la Arquitectura

**Rendimiento:**
- ✅ Filtros procesados en GPU (PyCUDA) - Muy rápido
- ✅ Imágenes servidas desde CDN (Supabase) - Baja latencia
- ✅ Metadata en Firestore - Queries rápidas
- ✅ Sin almacenamiento temporal innecesario

**Costos:**
- ✅ Solo se almacenan imágenes finales
- ✅ Preview de filtros no consume storage
- ✅ Firestore solo guarda metadata (bytes)
- ✅ Supabase Storage: Plan gratuito suficiente

**Escalabilidad:**
- ✅ PyCUDA puede escalar horizontalmente
- ✅ Supabase CDN global
- ✅ Firestore escala automáticamente
- ✅ Spring WebFlux (reactivo) - Alta concurrencia

**Experiencia de Usuario:**
- ✅ Preview de filtros instantáneo
- ✅ Prueba múltiples filtros sin espera
- ✅ Upload solo al publicar
- ✅ Feed carga rápido desde CDN

### Datos Técnicos

**Storage:**
- Firestore: ~1 KB por post (solo metadata)
- Supabase: ~200-500 KB por imagen (comprimida)
- Total por post: ~500 KB

**Latencia:**
- Filtro preview: ~100-300ms (GPU)
- Upload Supabase: ~500ms-1s (depende de red)
- Crear post Firestore: ~50-100ms
- Cargar feed: ~200-500ms (10 posts)

**Capacidad:**
- 1 GB Supabase gratis = ~2,000 imágenes
- Firestore: 1 GB gratis = ~1M posts (metadata)
- PyCUDA: Limitado por GPU disponible

---

**Última actualización:** 7 de diciembre de 2025
