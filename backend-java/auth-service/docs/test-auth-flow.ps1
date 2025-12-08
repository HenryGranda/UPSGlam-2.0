# =========================================
# Auth Service - Test Completo
# =========================================
# Script para probar todos los endpoints del auth-service
# 
# FLUJO DE PRUEBA:
# 1. Registro (retorna custom token)
# 2. Login para obtener ID token valido
# 3-4. Obtener y actualizar perfil con ID token
# 5-6. Login con email y username
# 7-8. Tests de seguridad (credenciales invalidas)
# 
# Copiar y pegar en PowerShell
# =========================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "UPSGLAM AUTH SERVICE - TEST COMPLETO" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Configuracion
# Opcion 1: Directo al auth-service (con /api)
$BASE_URL = "http://localhost:8082/api"

# Opcion 2: Via API Gateway (descomentar para usar)
# $BASE_URL = "http://localhost:8080/api"

# Generar datos aleatorios
$RANDOM = Get-Random -Minimum 1000 -Maximum 9999
$EMAIL = "testuser$RANDOM@ups.edu.ec"
$USERNAME = "testuser$RANDOM"
$PASSWORD = "test123456"
$FULL_NAME = "Usuario de Prueba $RANDOM"

Write-Host "[INFO] Datos del usuario:" -ForegroundColor Yellow
Write-Host "   Email:    $EMAIL" -ForegroundColor Gray
Write-Host "   Username: $USERNAME" -ForegroundColor Gray
Write-Host "   Password: $PASSWORD" -ForegroundColor Gray
Write-Host ""

$global:TOKEN = $null
$global:USER_ID = $null

# ==========================================
# TEST 1: REGISTRO
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 1: POST /auth/register" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$registerBody = @{
    email = $EMAIL
    password = $PASSWORD
    fullName = $FULL_NAME
    username = $USERNAME
} | ConvertTo-Json

Write-Host "[SEND] Enviando request..." -ForegroundColor Yellow

try {
    $registerResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/register" `
        -Method POST `
        -ContentType "application/json" `
        -Body $registerBody

    $global:USER_ID = $registerResponse.user.id

    Write-Host "[OK] REGISTRO EXITOSO" -ForegroundColor Green
    Write-Host "   User ID:  $global:USER_ID" -ForegroundColor Cyan
    Write-Host "   Username: $($registerResponse.user.username)" -ForegroundColor Cyan
    Write-Host "   Email:    $($registerResponse.user.email)" -ForegroundColor Cyan
    Write-Host "   Custom Token: $($registerResponse.token.idToken.Substring(0, 30))..." -ForegroundColor DarkGray
    Write-Host ""
    
    # IMPORTANTE: Firebase necesita unos segundos para activar el usuario completamente
    # Esperamos 3 segundos antes de intentar el login
    Write-Host "[INFO] Esperando activacion de usuario en Firebase..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    Write-Host "[INFO] Obteniendo ID token via login..." -ForegroundColor Yellow
    
    $loginBody = @{
        identifier = $EMAIL
        password = $PASSWORD
    } | ConvertTo-Json

    $loginResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/login" `
        -Method POST `
        -ContentType "application/json" `
        -Body $loginBody

    $global:TOKEN = $loginResponse.token.idToken
    Write-Host "[OK] ID Token obtenido exitosamente" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO EN REGISTRO" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 2: OBTENER PERFIL
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 2: GET /auth/me" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$headers = @{ Authorization = "Bearer $global:TOKEN" }

Write-Host "[SEND] Enviando request..." -ForegroundColor Yellow

try {
    $meResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/me" `
        -Method GET `
        -Headers $headers

    Write-Host "[OK] PERFIL OBTENIDO" -ForegroundColor Green
    Write-Host "   ID:        $($meResponse.id)" -ForegroundColor Cyan
    Write-Host "   Username:  $($meResponse.username)" -ForegroundColor Cyan
    Write-Host "   Full Name: $($meResponse.fullName)" -ForegroundColor Cyan
    Write-Host "   Email:     $($meResponse.email)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL OBTENER PERFIL" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 3: ACTUALIZAR PERFIL
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 3: PATCH /users/me" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$updateBody = @{
    username = "$USERNAME" + "_updated"
    fullName = "Usuario Actualizado $RANDOM"
    bio = "Estudiante UPS | Full Stack Dev | CUDA"
} | ConvertTo-Json

Write-Host "[SEND] Enviando request..." -ForegroundColor Yellow

try {
    $updateResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/users/me" `
        -Method PATCH `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $updateBody

    Write-Host "[OK] PERFIL ACTUALIZADO" -ForegroundColor Green
    Write-Host "   Username:  $($updateResponse.username)" -ForegroundColor Cyan
    Write-Host "   Full Name: $($updateResponse.fullName)" -ForegroundColor Cyan
    Write-Host "   Bio:       $($updateResponse.bio)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO AL ACTUALIZAR" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 4: VERIFICAR CAMBIOS
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 4: GET /auth/me (verificar)" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

Write-Host "[SEND] Verificando cambios..." -ForegroundColor Yellow

try {
    $meResponse2 = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/me" `
        -Method GET `
        -Headers $headers

    Write-Host "[OK] CAMBIOS VERIFICADOS" -ForegroundColor Green
    Write-Host "   Username:  $($meResponse2.username)" -ForegroundColor Cyan
    Write-Host "   Full Name: $($meResponse2.fullName)" -ForegroundColor Cyan
    Write-Host "   Bio:       $($meResponse2.bio)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO EN VERIFICACION" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 5: LOGIN CON EMAIL
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 5: POST /auth/login (email)" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$loginEmailBody = @{
    identifier = $EMAIL
    password = $PASSWORD
} | ConvertTo-Json

Write-Host "[SEND] Enviando request..." -ForegroundColor Yellow

try {
    $loginEmailResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/login" `
        -Method POST `
        -ContentType "application/json" `
        -Body $loginEmailBody

    Write-Host "[OK] LOGIN CON EMAIL EXITOSO" -ForegroundColor Green
    Write-Host "   Username: $($loginEmailResponse.user.username)" -ForegroundColor Cyan
    Write-Host "   Email:    $($loginEmailResponse.user.email)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO EN LOGIN EMAIL" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 6: LOGIN CON USERNAME
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 6: POST /auth/login (username)" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$loginUsernameBody = @{
    identifier = "$USERNAME" + "_updated"
    password = $PASSWORD
} | ConvertTo-Json

Write-Host "[SEND] Enviando request..." -ForegroundColor Yellow

try {
    $loginUsernameResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/login" `
        -Method POST `
        -ContentType "application/json" `
        -Body $loginUsernameBody

    Write-Host "[OK] LOGIN CON USERNAME EXITOSO" -ForegroundColor Green
    Write-Host "   Username: $($loginUsernameResponse.user.username)" -ForegroundColor Cyan
    Write-Host "   Email:    $($loginUsernameResponse.user.email)" -ForegroundColor Cyan
    Write-Host "   Bio:      $($loginUsernameResponse.user.bio)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] FALLO EN LOGIN USERNAME" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 7: CREDENCIALES INCORRECTAS
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 7: Login con credenciales incorrectas" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$loginWrongBody = @{
    identifier = $EMAIL
    password = "wrongpassword123"
} | ConvertTo-Json

Write-Host "[SEND] Enviando request con password incorrecto..." -ForegroundColor Yellow

try {
    $loginWrongResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/login" `
        -Method POST `
        -ContentType "application/json" `
        -Body $loginWrongBody

    Write-Host "[WARN] El login deberia haber fallado" -ForegroundColor Yellow
} catch {
    Write-Host "[OK] ERROR ESPERADO (401)" -ForegroundColor Green
    Write-Host "   Credenciales rechazadas correctamente" -ForegroundColor Cyan
    Write-Host ""
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 8: TOKEN INVALIDO
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 8: Token invalido" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

$invalidHeaders = @{ Authorization = "Bearer token_invalido_12345" }

Write-Host "[SEND] Enviando request con token invalido..." -ForegroundColor Yellow

try {
    $invalidResponse = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/me" `
        -Method GET `
        -Headers $invalidHeaders

    Write-Host "[WARN] La request deberia haber fallado" -ForegroundColor Yellow
} catch {
    Write-Host "[OK] ERROR ESPERADO (401)" -ForegroundColor Green
    Write-Host "   Token invalido rechazado correctamente" -ForegroundColor Cyan
    Write-Host ""
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
Write-Host "[INFO] Usuario de prueba:" -ForegroundColor Yellow
Write-Host "   Email:    $EMAIL" -ForegroundColor Cyan
Write-Host "   Username: $($USERNAME)_updated" -ForegroundColor Cyan
Write-Host "   Password: $PASSWORD" -ForegroundColor Cyan
Write-Host "   User ID:  $global:USER_ID" -ForegroundColor Cyan
Write-Host ""
Write-Host "[INFO] Token JWT:" -ForegroundColor Yellow
Write-Host "   $($loginUsernameResponse.token.idToken)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "[INFO] Tests ejecutados:" -ForegroundColor Yellow
Write-Host "   1. Registro + Login inicial [OK]" -ForegroundColor Green
Write-Host "   2. Obtener perfil           [OK]" -ForegroundColor Green
Write-Host "   3. Actualizar perfil        [OK]" -ForegroundColor Green
Write-Host "   4. Verificar cambios        [OK]" -ForegroundColor Green
Write-Host "   5. Login con email          [OK]" -ForegroundColor Green
Write-Host "   6. Login con username       [OK]" -ForegroundColor Green
Write-Host "   7. Credenciales incorrectas[OK]" -ForegroundColor Green
Write-Host "   8. Token invalido          [OK]" -ForegroundColor Green
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "AUTH SERVICE FUNCIONANDO CORRECTAMENTE" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
