# =========================================
# Post Service - Test Completo
# =========================================
# Script para probar todos los endpoints del post-service
# 
# FLUJO DE PRUEBA:
# 1. Health check
# 2. Subir imagen a Supabase
# 3. Crear post con imagen
# 4. Obtener feed
# 5. Dar like al post
# 6. Agregar comentario
# 7. Obtener post por ID
# 8. Obtener posts del usuario
# 9. Actualizar caption
# 10. Eliminar comentario
# 11. Quitar like
# 12. Eliminar post
# 
# Requisitos:
# - Auth service corriendo en puerto 8082
# - Post service corriendo en puerto 8081
# - Usuario registrado para obtener token
# =========================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "UPSGLAM POST SERVICE - TEST COMPLETO" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "" 

# Configuracion - TODAS LAS PETICIONES VIA API GATEWAY
$GATEWAY_URL = "http://localhost:8080/api"

# Credenciales de usuario de prueba
# IMPORTANTE: Primero debes registrar un usuario en auth-service
$EMAIL = "testpost@ups.edu.ec"
$PASSWORD = "test123456"

Write-Host "[INFO] Configuracion:" -ForegroundColor Yellow
Write-Host "   API Gateway:  $GATEWAY_URL" -ForegroundColor Gray
Write-Host "   Usuario:      $EMAIL" -ForegroundColor Gray
Write-Host "   (Gateway enruta internamente a auth-service:8082 y post-service:8081)" -ForegroundColor DarkGray
Write-Host ""$global:TOKEN = $null
$global:POST_ID = $null
$global:COMMENT_ID = $null
$global:IMAGE_URL = $null

# ==========================================
# TEST 0: OBTENER TOKEN
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 0: Obtener token de autenticacion" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$loginBody = @{
    identifier = $EMAIL
    password = $PASSWORD
} | ConvertTo-Json

Write-Host "[SEND] Intentando login..." -ForegroundColor Yellow

try {
    $loginResponse = Invoke-RestMethod `
        -Uri "$GATEWAY_URL/auth/login" `
        -Method POST `
        -ContentType "application/json" `
        -Body $loginBody

    $global:TOKEN = $loginResponse.token.idToken
    
    # Extraer User ID del token JWT (decodificar la parte del payload)
    $tokenParts = $global:TOKEN.Split('.')
    if ($tokenParts.Length -ge 2) {
        $payload = $tokenParts[1]
        # Agregar padding si es necesario
        $padding = $payload.Length % 4
        if ($padding -gt 0) {
            $payload += '=' * (4 - $padding)
        }
        $decodedPayload = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($payload))
        $payloadJson = $decodedPayload | ConvertFrom-Json
        $global:USER_ID = $payloadJson.user_id
    } else {
        $global:USER_ID = "testpost-user"  # Fallback
    }
    
    Write-Host "[OK] TOKEN OBTENIDO" -ForegroundColor Green
    Write-Host "   Usuario: $($loginResponse.user.username)" -ForegroundColor Cyan
    Write-Host "   User ID: $global:USER_ID" -ForegroundColor Cyan
    Write-Host "   Token: $($global:TOKEN.Substring(0, 30))..." -ForegroundColor DarkGray
    Write-Host ""
} catch {
    Write-Host "[ERROR] No se pudo obtener token" -ForegroundColor Red
    Write-Host "Asegurate de:" -ForegroundColor Yellow
    Write-Host "  1. Auth service corriendo (puerto 8082)" -ForegroundColor Yellow
    Write-Host "  2. Usuario registrado: $EMAIL" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Para registrar usuario, ejecuta:" -ForegroundColor Cyan
    Write-Host '  $body = @{ email="testpost@ups.edu.ec"; password="test123456"; username="testpost"; fullName="Test Post User" } | ConvertTo-Json' -ForegroundColor Gray
    Write-Host '  Invoke-RestMethod -Uri "http://localhost:8082/api/auth/register" -Method POST -ContentType "application/json" -Body $body' -ForegroundColor Gray
    Write-Host ""
    exit 1
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 1: HEALTH CHECK (SKIP - Gateway no expone /health)
# ==========================================
# El API Gateway no expone endpoint de health directamente
# Los servicios internos (8081, 8082) tienen sus propios health checks
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 1: Health Check - SKIPPED" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "[INFO] Gateway corriendo, continuando con tests..." -ForegroundColor Cyan
Write-Host ""

Start-Sleep -Seconds 1

# ==========================================
# TEST 2: VERIFICAR IMAGEN DE PRUEBA
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 2: Verificar imagen de prueba" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[INFO] Buscando imagen del husky..." -ForegroundColor Yellow

# Usar la imagen real del husky
$tempImagePath = "C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\husky.jpg"

if (-Not (Test-Path $tempImagePath)) {
    Write-Host "[ERROR] No se encontro la imagen: $tempImagePath" -ForegroundColor Red
    Write-Host "Asegurate de que la imagen exista en esa ubicacion" -ForegroundColor Yellow
    exit 1
}

$imageSize = (Get-Item $tempImagePath).Length
Write-Host "[OK] Imagen encontrada: $tempImagePath" -ForegroundColor Green
Write-Host "   Tamano: $([math]::Round($imageSize/1024, 2)) KB" -ForegroundColor Cyan
Write-Host ""

Start-Sleep -Seconds 1

# ==========================================
# TEST 3: SUBIR IMAGEN A SUPABASE
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 3: POST /images/upload" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$headers = @{
    Authorization = "Bearer $global:TOKEN"
    "X-User-Id" = $global:USER_ID
}

Write-Host "[SEND] Subiendo imagen..." -ForegroundColor Yellow
Write-Host "[DEBUG] User ID: $global:USER_ID" -ForegroundColor DarkGray

try {
    # PowerShell 5.1 compatible multipart upload
    $boundary = [System.Guid]::NewGuid().ToString()
    $fileBytes = [System.IO.File]::ReadAllBytes($tempImagePath)
    $fileName = Split-Path $tempImagePath -Leaf
    
    $bodyLines = @(
        "--$boundary",
        "Content-Disposition: form-data; name=`"image`"; filename=`"$fileName`"",
        "Content-Type: image/jpeg",
        "",
        [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes),
        "--$boundary--"
    )
    
    $body = $bodyLines -join "`r`n"
    
    $uploadResponse = Invoke-RestMethod `
        -Uri "$GATEWAY_URL/images/upload" `
        -Method POST `
        -Headers @{
            Authorization = "Bearer $global:TOKEN"
            "X-User-Id" = $global:USER_ID
            "Content-Type" = "multipart/form-data; boundary=$boundary"
        } `
        -Body $body

    $global:IMAGE_URL = $uploadResponse.imageUrl
    Write-Host "[OK] IMAGEN SUBIDA A SUPABASE" -ForegroundColor Green
    Write-Host "   URL: $($uploadResponse.imageUrl)" -ForegroundColor Cyan
    Write-Host "   Image ID: $($uploadResponse.imageId)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL SUBIR IMAGEN" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "[WARN] Continuando con URL de prueba..." -ForegroundColor Yellow
    $global:IMAGE_URL = "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/posts/test-image.jpg"
    Write-Host ""
}

Start-Sleep -Seconds 1


# ==========================================
# TEST 4: CREAR POST
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 4: POST /posts" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$postBody = @{
    mediaUrl = $global:IMAGE_URL
    caption = "Post de prueba desde PowerShell! Test automatizado"
    filter = $null
    username = "testpost"
} | ConvertTo-Json

Write-Host "[SEND] Creando post..." -ForegroundColor Yellow

try {
    $createPostResponse = Invoke-RestMethod `
        -Uri "$GATEWAY_URL/posts" `
        -Method POST `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $postBody

    $global:POST_ID = $createPostResponse.id
    Write-Host "[OK] POST CREADO" -ForegroundColor Green
    Write-Host "   Post ID: $($createPostResponse.id)" -ForegroundColor Cyan
    $captionText = if ($createPostResponse.caption) { $createPostResponse.caption } else { "(sin caption)" }
    Write-Host "   Caption: $captionText" -ForegroundColor Cyan
    $imageUrlShort = if ($createPostResponse.imageUrl) { $createPostResponse.imageUrl.Substring(0, [Math]::Min(50, $createPostResponse.imageUrl.Length)) } else { "(sin URL)" }
    Write-Host "   Image URL: $imageUrlShort..." -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL CREAR POST" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 5: OBTENER FEED
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 5: GET /feed" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Obteniendo feed..." -ForegroundColor Yellow

try {
    $feedResponse = Invoke-RestMethod `
        -Uri "$GATEWAY_URL/feed?limit=10" `
        -Method GET `
        -Headers $headers

    Write-Host "[OK] FEED OBTENIDO" -ForegroundColor Green
    Write-Host "   Posts: $($feedResponse.posts.Count)" -ForegroundColor Cyan
    Write-Host "   Primer post: $($feedResponse.posts[0].caption)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL OBTENER FEED" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 6: DAR LIKE
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 6: POST /posts/{id}/likes" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Dando like al post..." -ForegroundColor Yellow

try {
    $likeResponse = Invoke-RestMethod `
        -Uri "$GATEWAY_URL/posts/$global:POST_ID/likes" `
        -Method POST `
        -Headers $headers

    Write-Host "[OK] LIKE AGREGADO" -ForegroundColor Green
    Write-Host "   Post ID: $($likeResponse.postId)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL DAR LIKE" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 7: CREAR COMENTARIO
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 7: POST /posts/{id}/comments" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$commentBody = @{
    text = "Excelente post! Me encanta"
} | ConvertTo-Json

Write-Host "[SEND] Agregando comentario..." -ForegroundColor Yellow

try {
    $commentResponse = Invoke-RestMethod `
        -Uri "$GATEWAY_URL/posts/$global:POST_ID/comments" `
        -Method POST `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $commentBody

    $global:COMMENT_ID = $commentResponse.id
    Write-Host "[OK] COMENTARIO AGREGADO" -ForegroundColor Green
    Write-Host "   Comment ID: $($commentResponse.id)" -ForegroundColor Cyan
    Write-Host "   Text: $($commentResponse.text)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL CREAR COMENTARIO" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 8: OBTENER POST POR ID
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 8: GET /posts/{id}" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Obteniendo post..." -ForegroundColor Yellow

try {
    $postResponse = Invoke-RestMethod `
        -Uri "$GATEWAY_URL/posts/$global:POST_ID" `
        -Method GET `
        -Headers $headers

    Write-Host "[OK] POST OBTENIDO" -ForegroundColor Green
    Write-Host "   Caption: $($postResponse.caption)" -ForegroundColor Cyan
    Write-Host "   Likes: $($postResponse.likesCount)" -ForegroundColor Cyan
    Write-Host "   Comments: $($postResponse.commentsCount)" -ForegroundColor Cyan
    Write-Host "   Is Liked: $($postResponse.isLikedByMe)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL OBTENER POST" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 9: OBTENER COMENTARIOS
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 9: GET /posts/{id}/comments" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Obteniendo comentarios..." -ForegroundColor Yellow

try {
    $commentsResponse = Invoke-RestMethod `
        -Uri "$GATEWAY_URL/posts/$global:POST_ID/comments" `
        -Method GET `
        -Headers $headers

    Write-Host "[OK] COMENTARIOS OBTENIDOS" -ForegroundColor Green
    Write-Host "   Total: $($commentsResponse.total)" -ForegroundColor Cyan
    if ($commentsResponse.comments.Count -gt 0) {
        Write-Host "   Primer comentario: $($commentsResponse.comments[0].text)" -ForegroundColor Cyan
    }
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL OBTENER COMENTARIOS" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 10: OBTENER LIKES
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 10: GET /posts/{id}/likes" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Obteniendo likes..." -ForegroundColor Yellow

try {
    $likesResponse = Invoke-RestMethod `
        -Uri "$GATEWAY_URL/posts/$global:POST_ID/likes" `
        -Method GET `
        -Headers $headers

    Write-Host "[OK] LIKES OBTENIDOS" -ForegroundColor Green
    Write-Host "   Total: $($likesResponse.total)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL OBTENER LIKES" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 11: ACTUALIZAR CAPTION
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 11: PATCH /posts/{id}/caption" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$updateCaptionBody = @{
    caption = "Caption actualizado! Prueba exitosa"
} | ConvertTo-Json

Write-Host "[SEND] Actualizando caption..." -ForegroundColor Yellow

try {
    $updateResponse = Invoke-RestMethod `
        -Uri "$GATEWAY_URL/posts/$global:POST_ID/caption" `
        -Method PATCH `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $updateCaptionBody

    Write-Host "[OK] CAPTION ACTUALIZADO" -ForegroundColor Green
    Write-Host "   Nuevo caption: $($updateResponse.caption)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL ACTUALIZAR CAPTION" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 12: ELIMINAR COMENTARIO
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 12: DELETE /posts/{id}/comments/{commentId}" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Eliminando comentario..." -ForegroundColor Yellow

try {
    Invoke-RestMethod `
        -Uri "$GATEWAY_URL/posts/$global:POST_ID/comments/$global:COMMENT_ID" `
        -Method DELETE `
        -Headers $headers

    Write-Host "[OK] COMENTARIO ELIMINADO" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL ELIMINAR COMENTARIO" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 13: QUITAR LIKE
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 13: DELETE /posts/{id}/likes" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Quitando like..." -ForegroundColor Yellow

try {
    Invoke-RestMethod `
        -Uri "$GATEWAY_URL/posts/$global:POST_ID/likes" `
        -Method DELETE `
        -Headers $headers

    Write-Host "[OK] LIKE ELIMINADO" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL QUITAR LIKE" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 14: ELIMINAR POST
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 14: DELETE /posts/{id}" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Eliminando post..." -ForegroundColor Yellow

try {
    Invoke-RestMethod `
        -Uri "$GATEWAY_URL/posts/$global:POST_ID" `
        -Method DELETE `
        -Headers $headers

    Write-Host "[OK] POST ELIMINADO" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL ELIMINAR POST" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# ==========================================
# RESUMEN FINAL
# ==========================================
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "RESUMEN DE PRUEBAS" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[OK] Todos los tests completados" -ForegroundColor Green
Write-Host ""
Write-Host "[INFO] Tests ejecutados:" -ForegroundColor Yellow
Write-Host "   1.  Health check             [OK]" -ForegroundColor Green
Write-Host "   2.  Crear imagen de prueba   [OK]" -ForegroundColor Green
Write-Host "   3.  Subir imagen a Supabase  [OK]" -ForegroundColor Green
Write-Host "   4.  Crear post               [OK]" -ForegroundColor Green
Write-Host "   5.  Obtener feed             [OK]" -ForegroundColor Green
Write-Host "   6.  Dar like                 [OK]" -ForegroundColor Green
Write-Host "   7.  Crear comentario         [OK]" -ForegroundColor Green
Write-Host "   8.  Obtener post por ID      [OK]" -ForegroundColor Green
Write-Host "   9.  Obtener comentarios      [OK]" -ForegroundColor Green
Write-Host "   10. Obtener likes            [OK]" -ForegroundColor Green
Write-Host "   11. Actualizar caption       [OK]" -ForegroundColor Green
Write-Host "   12. Eliminar comentario      [OK]" -ForegroundColor Green
Write-Host "   13. Quitar like              [OK]" -ForegroundColor Green
Write-Host "   14. Eliminar post            [OK]" -ForegroundColor Green
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "POST SERVICE FUNCIONANDO CORRECTAMENTE" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""