# =========================================
# Test: Flujo completo CON FILTRO de PyCUDA
# =========================================
# FLUJO:
# 1. Login → Obtener token
# 2. Subir imagen a POST /images/preview con filtro
#    → Post-service llama a PyCUDA (puerto 5000)
#    → PyCUDA aplica filtro CUDA
#    → Post-service sube a Supabase temp/
#    → Devuelve tempImageId + URL temporal
# 3. Crear post con POST /posts usando tempImageId
#    → Post-service mueve imagen de temp/ a posts/
#    → Guarda post en Firestore con URL final
# =========================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "UPSGLAM - TEST FLUJO CON FILTRO PYCUDA" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Configuracion - TODO VIA API GATEWAY
$BASE_URL = "http://localhost:8080/api"
$BASE_URL_CUDA = "http://localhost:5000"

# Credenciales de usuario de prueba
$EMAIL = "testpost@ups.edu.ec"
$PASSWORD = "test123456"

# Imagen de prueba
$IMAGE_PATH = "C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\husky.jpg"

Write-Host "[INFO] Configuracion:" -ForegroundColor Yellow
Write-Host "   API Gateway: $BASE_URL" -ForegroundColor Gray
Write-Host "   CUDA Service: $BASE_URL_CUDA (directo, interno)" -ForegroundColor Gray
Write-Host "   Usuario:      $EMAIL" -ForegroundColor Gray
Write-Host "   Imagen:       $IMAGE_PATH" -ForegroundColor Gray
Write-Host ""

$global:TOKEN = $null
$global:USER_ID = $null
$global:TEMP_IMAGE_ID = $null
$global:TEMP_IMAGE_URL = $null
$global:POST_ID = $null

# ==========================================
# PASO 1: LOGIN Y OBTENER TOKEN
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "PASO 1: Obtener token de autenticacion" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$loginBody = @{
    identifier = $EMAIL
    password = $PASSWORD
} | ConvertTo-Json

Write-Host "[SEND] Intentando login..." -ForegroundColor Yellow

try {
    $loginResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/login" `
        -Method POST `
        -ContentType "application/json" `
        -Body $loginBody

    $global:TOKEN = $loginResponse.token.idToken
    
    # Extraer User ID del token JWT
    $tokenParts = $global:TOKEN.Split('.')
    if ($tokenParts.Length -ge 2) {
        $payload = $tokenParts[1]
        $padding = $payload.Length % 4
        if ($padding -gt 0) {
            $payload += '=' * (4 - $padding)
        }
        $decodedPayload = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($payload))
        $payloadJson = $decodedPayload | ConvertFrom-Json
        $global:USER_ID = $payloadJson.user_id
    } else {
        $global:USER_ID = "testpost-user"
    }
    
    Write-Host "[OK] TOKEN OBTENIDO" -ForegroundColor Green
    Write-Host "   Usuario: $($loginResponse.user.username)" -ForegroundColor Cyan
    Write-Host "   User ID: $global:USER_ID" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] No se pudo obtener token" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Start-Sleep -Seconds 1

# ==========================================
# PASO 2: VERIFICAR SERVICIOS
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "PASO 2: Verificar servicios activos" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

# Verificar PyCUDA
Write-Host "[CHECK] Verificando PyCUDA Service..." -ForegroundColor Yellow
try {
    $cudaHealth = Invoke-RestMethod -Uri "$BASE_URL_CUDA/health" -Method GET
    Write-Host "[OK] PyCUDA Service UP" -ForegroundColor Green
    Write-Host "   Status: $($cudaHealth.status)" -ForegroundColor Cyan
} catch {
    Write-Host "[ERROR] PyCUDA Service no disponible" -ForegroundColor Red
    Write-Host "Inicia el servicio con: python -m uvicorn app:app --host 0.0.0.0 --port 5000 --reload" -ForegroundColor Yellow
    exit 1
}

# Verificar Post Service
Write-Host "[CHECK] Verificando Post Service..." -ForegroundColor Yellow
try {
    $postHealth = Invoke-RestMethod -Uri "http://localhost:8081/api/health" -Method GET
    Write-Host "[OK] Post Service UP" -ForegroundColor Green
    Write-Host "   Status: $($postHealth.status)" -ForegroundColor Cyan
} catch {
    Write-Host "[ERROR] Post Service no disponible en puerto 8081" -ForegroundColor Red
    exit 1
}

# Verificar imagen
Write-Host "[CHECK] Verificando imagen..." -ForegroundColor Yellow
if (-Not (Test-Path $IMAGE_PATH)) {
    Write-Host "[ERROR] No se encontro la imagen: $IMAGE_PATH" -ForegroundColor Red
    exit 1
}
$imageSize = (Get-Item $IMAGE_PATH).Length
Write-Host "[OK] Imagen encontrada" -ForegroundColor Green
Write-Host "   Tamano: $([math]::Round($imageSize/1024, 2)) KB" -ForegroundColor Cyan
Write-Host ""

Start-Sleep -Seconds 1

# ==========================================
# PASO 3: APLICAR FILTRO (PyCUDA)
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "PASO 3: POST /images/preview (con filtro)" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

# Filtros disponibles: gaussian, box_blur, prewitt, laplacian, ups_logo, ups_color
$FILTER = "gaussian"

Write-Host "[SEND] Subiendo imagen con filtro: $FILTER" -ForegroundColor Yellow
Write-Host "[INFO] Flujo interno:" -ForegroundColor Gray
Write-Host "   1. Post-service recibe imagen" -ForegroundColor Gray
Write-Host "   2. Post-service → PyCUDA (puerto 5000)" -ForegroundColor Gray
Write-Host "   3. PyCUDA aplica filtro CUDA" -ForegroundColor Gray
Write-Host "   4. Post-service sube a Supabase temp/" -ForegroundColor Gray
Write-Host "   5. Devuelve tempImageId + URL" -ForegroundColor Gray
Write-Host ""

try {
    # PowerShell 5.1 compatible multipart upload
    $boundary = [System.Guid]::NewGuid().ToString()
    $fileBytes = [System.IO.File]::ReadAllBytes($IMAGE_PATH)
    $fileName = Split-Path $IMAGE_PATH -Leaf
    
    $bodyLines = @(
        "--$boundary",
        "Content-Disposition: form-data; name=`"image`"; filename=`"$fileName`"",
        "Content-Type: image/jpeg",
        "",
        [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes),
        "--$boundary",
        "Content-Disposition: form-data; name=`"filter`"",
        "",
        $FILTER,
        "--$boundary--"
    )
    
    $body = $bodyLines -join "`r`n"
    
    $previewResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/images/preview" `
        -Method POST `
        -Headers @{
            Authorization = "Bearer $global:TOKEN"
            "X-User-Id" = $global:USER_ID
            "Content-Type" = "multipart/form-data; boundary=$boundary"
        } `
        -Body $body

    $global:TEMP_IMAGE_ID = $previewResponse.tempImageId
    $global:TEMP_IMAGE_URL = $previewResponse.imageUrl
    
    Write-Host "[OK] IMAGEN PROCESADA CON FILTRO" -ForegroundColor Green
    Write-Host "   Filtro aplicado: $($previewResponse.filter)" -ForegroundColor Cyan
    Write-Host "   Temp Image ID: $global:TEMP_IMAGE_ID" -ForegroundColor Cyan
    Write-Host "   URL temporal: $global:TEMP_IMAGE_URL" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "[INFO] En la app mobile, el usuario previsualiza esta imagen" -ForegroundColor Yellow
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL PROCESAR IMAGEN CON FILTRO" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "[DEBUG] Verifica que PyCUDA este corriendo en puerto 5000" -ForegroundColor Yellow
    exit 1
}

Start-Sleep -Seconds 2

# ==========================================
# PASO 4: CREAR POST (con tempImageId)
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "PASO 4: POST /posts (crear post)" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Creando post con imagen filtrada..." -ForegroundColor Yellow
Write-Host "[INFO] Flujo interno:" -ForegroundColor Gray
Write-Host "   1. Post-service recibe tempImageId" -ForegroundColor Gray
Write-Host "   2. Post-service mueve imagen de temp/ a posts/" -ForegroundColor Gray
Write-Host "   3. Post-service guarda en Firestore con URL final" -ForegroundColor Gray
Write-Host ""

$postBody = @{
    tempImageId = $global:TEMP_IMAGE_URL
    filter = $FILTER
    caption = "Foto con filtro $FILTER aplicado con CUDA! #UPSGlam #CUDA"
    username = "testpost"
} | ConvertTo-Json -Depth 10

try {
    $postResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/posts" `
        -Method POST `
        -Headers @{
            Authorization = "Bearer $global:TOKEN"
            "X-User-Id" = $global:USER_ID
            "Content-Type" = "application/json; charset=utf-8"
        } `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($postBody))

    $global:POST_ID = $postResponse.id
    
    Write-Host "[OK] POST CREADO EXITOSAMENTE" -ForegroundColor Green
    Write-Host "   Post ID: $($postResponse.id)" -ForegroundColor Cyan
    Write-Host "   Usuario: $($postResponse.username)" -ForegroundColor Cyan
    Write-Host "   Filtro: $($postResponse.filter)" -ForegroundColor Cyan
    Write-Host "   Caption: $($postResponse.description)" -ForegroundColor Cyan
    Write-Host "   Imagen URL: $($postResponse.imageUrl)" -ForegroundColor Cyan
    Write-Host "   Likes: $($postResponse.likesCount)" -ForegroundColor Cyan
    Write-Host "   Comentarios: $($postResponse.commentsCount)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL CREAR POST" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Start-Sleep -Seconds 1

# ==========================================
# PASO 5: VERIFICAR POST EN FEED
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "PASO 5: GET /feed (verificar post)" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Obteniendo feed..." -ForegroundColor Yellow

try {
    $feedResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/feed?limit=10" `
        -Method GET `
        -Headers @{
            Authorization = "Bearer $global:TOKEN"
            "X-User-Id" = $global:USER_ID
        }

    $myPost = $feedResponse.posts | Where-Object { $_.id -eq $global:POST_ID }
    
    if ($myPost) {
        Write-Host "[OK] POST ENCONTRADO EN FEED" -ForegroundColor Green
        Write-Host "   Posicion en feed: $($feedResponse.posts.IndexOf($myPost) + 1) de $($feedResponse.posts.Count)" -ForegroundColor Cyan
        Write-Host ""
    } else {
        Write-Host "[WARN] Post no encontrado en feed (puede tardar en aparecer)" -ForegroundColor Yellow
        Write-Host ""
    }
} catch {
    Write-Host "[ERROR] FALLO AL OBTENER FEED" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# ==========================================
# RESUMEN FINAL
# ==========================================
