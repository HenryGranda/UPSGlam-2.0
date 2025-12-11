# =========================================
# Script de prueba completo: Login + Follows
# =========================================
# Este script primero hace login con dos usuarios reales
# y luego prueba todos los endpoints de follows
# =========================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "PRUEBA COMPLETA: LOGIN + FOLLOWS" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Configuracion - Via API Gateway
$BASE_URL = "http://localhost:8080/api"
# Para pruebas directas cambiar a: http://localhost:8082/api

# Usuarios reales
$USER1_EMAIL = "hgrandal@est.ups.edu.ec"
$USER1_PASS = "huambi123"

$USER2_EMAIL = "jgrandal@est.ups.edu.ec"  # Asumiendo que el email tiene este formato
$USER2_PASS = "sucua123"

$global:TOKEN_USER1 = $null
$global:TOKEN_USER2 = $null
$global:USER1_ID = $null
$global:USER2_ID = $null

# ==========================================
# PASO 1: LOGIN USER 1
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "PASO 1: Login Usuario 1" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "Email: $USER1_EMAIL" -ForegroundColor Gray

$loginBody1 = @{
    identifier = $USER1_EMAIL
    password = $USER1_PASS
} | ConvertTo-Json

try {
    $loginResponse1 = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/login" `
        -Method POST `
        -ContentType "application/json" `
        -Body $loginBody1

    $global:TOKEN_USER1 = $loginResponse1.token.idToken
    $global:USER1_ID = $loginResponse1.user.id

    Write-Host "[OK] Login exitoso" -ForegroundColor Green
    Write-Host "   User ID: $global:USER1_ID" -ForegroundColor Cyan
    Write-Host "   Username: $($loginResponse1.user.username)" -ForegroundColor Cyan
    Write-Host "   Token: $($global:TOKEN_USER1.Substring(0, 30))..." -ForegroundColor DarkGray
    Write-Host ""
} catch {
    Write-Host "[ERROR] Fallo en login User 1" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "[INFO] Intentando con username 'Jgranda'..." -ForegroundColor Yellow
    
    # Intentar con username en vez de email
    $loginBody1Alt = @{
        identifier = "Jgranda"
        password = $USER1_PASS
    } | ConvertTo-Json
    
    try {
        $loginResponse1 = Invoke-RestMethod `
            -Uri "$BASE_URL/auth/login" `
            -Method POST `
            -ContentType "application/json" `
            -Body $loginBody1Alt
        
        $global:TOKEN_USER1 = $loginResponse1.token.idToken
        $global:USER1_ID = $loginResponse1.user.id
        
        Write-Host "[OK] Login exitoso con username" -ForegroundColor Green
        Write-Host "   User ID: $global:USER1_ID" -ForegroundColor Cyan
        Write-Host ""
    } catch {
        Write-Host "[ERROR] No se pudo logear User 1" -ForegroundColor Red
        exit 1
    }
}

Start-Sleep -Seconds 1

# ==========================================
# PASO 2: LOGIN USER 2
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "PASO 2: Login Usuario 2" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "Probando con username: Jgranda" -ForegroundColor Gray

$loginBody2 = @{
    identifier = "Jgranda"
    password = $USER2_PASS
} | ConvertTo-Json

try {
    $loginResponse2 = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/login" `
        -Method POST `
        -ContentType "application/json" `
        -Body $loginBody2

    $global:TOKEN_USER2 = $loginResponse2.token.idToken
    $global:USER2_ID = $loginResponse2.user.id

    Write-Host "[OK] Login exitoso" -ForegroundColor Green
    Write-Host "   User ID: $global:USER2_ID" -ForegroundColor Cyan
    Write-Host "   Username: $($loginResponse2.user.username)" -ForegroundColor Cyan
    Write-Host "   Token: $($global:TOKEN_USER2.Substring(0, 30))..." -ForegroundColor DarkGray
    Write-Host ""
} catch {
    Write-Host "[ERROR] Fallo en login User 2" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "[INFO] Intentando con email..." -ForegroundColor Yellow
    
    $loginBody2Alt = @{
        identifier = $USER2_EMAIL
        password = $USER2_PASS
    } | ConvertTo-Json
    
    try {
        $loginResponse2 = Invoke-RestMethod `
            -Uri "$BASE_URL/auth/login" `
            -Method POST `
            -ContentType "application/json" `
            -Body $loginBody2Alt
        
        $global:TOKEN_USER2 = $loginResponse2.token.idToken
        $global:USER2_ID = $loginResponse2.user.id
        
        Write-Host "[OK] Login exitoso con email" -ForegroundColor Green
        Write-Host "   User ID: $global:USER2_ID" -ForegroundColor Cyan
        Write-Host ""
    } catch {
        Write-Host "[ERROR] No se pudo logear User 2" -ForegroundColor Red
        exit 1
    }
}

# Verificar que tenemos los tokens
if (-not $global:TOKEN_USER1 -or -not $global:TOKEN_USER2) {
    Write-Host "[ERROR] No se obtuvieron los tokens necesarios" -ForegroundColor Red
    exit 1
}

Start-Sleep -Seconds 1

Write-Host ""
Write-Host "=========================================" -ForegroundColor Magenta
Write-Host "PRUEBAS DE FOLLOWS" -ForegroundColor Magenta
Write-Host "=========================================" -ForegroundColor Magenta
Write-Host ""

# ==========================================
# TEST 1: User1 sigue a User2
# ==========================================
Write-Host "TEST 1: User1 sigue a User2" -ForegroundColor Yellow

$followBody = @{
    targetUserId = $global:USER2_ID
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/follows" `
        -Method POST `
        -Headers @{ "Authorization" = "Bearer $global:TOKEN_USER1" } `
        -ContentType "application/json" `
        -Body $followBody
    
    Write-Host "[OK] Follow exitoso" -ForegroundColor Green
    Write-Host "   isFollowing: $($response.isFollowing)" -ForegroundColor Cyan
    Write-Host "   followersCount: $($response.followersCount)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 2: Intentar seguir de nuevo (debe fallar)
# ==========================================
Write-Host "TEST 2: Intentar seguir de nuevo (debe fallar)" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/follows" `
        -Method POST `
        -Headers @{ "Authorization" = "Bearer $global:TOKEN_USER1" } `
        -ContentType "application/json" `
        -Body $followBody
    
    Write-Host "[FAIL] No deberia haber funcionado" -ForegroundColor Red
    Write-Host ""
} catch {
    Write-Host "[OK] Error esperado (ya sigue al usuario)" -ForegroundColor Green
    Write-Host ""
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 3: Ver estadisticas de User2
# ==========================================
Write-Host "TEST 3: Ver estadisticas de User2" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/follows/$global:USER2_ID/stats" `
        -Method GET `
        -Headers @{ "Authorization" = "Bearer $global:TOKEN_USER1" }
    
    Write-Host "[OK] Estadisticas obtenidas" -ForegroundColor Green
    Write-Host "   followersCount: $($response.followersCount)" -ForegroundColor Cyan
    Write-Host "   followingCount: $($response.followingCount)" -ForegroundColor Cyan
    Write-Host "   isFollowing: $($response.isFollowing)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 4: Ver estadisticas con listas
# ==========================================
Write-Host "TEST 4: Ver estadisticas con listas completas" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/follows/$global:USER2_ID/stats?includeList=true" `
        -Method GET `
        -Headers @{ "Authorization" = "Bearer $global:TOKEN_USER1" }
    
    Write-Host "[OK] Estadisticas con listas obtenidas" -ForegroundColor Green
    Write-Host "   followersCount: $($response.followersCount)" -ForegroundColor Cyan
    Write-Host "   followingCount: $($response.followingCount)" -ForegroundColor Cyan
    Write-Host "   Seguidores: $($response.followers.Count)" -ForegroundColor Cyan
    Write-Host "   Seguidos: $($response.following.Count)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 5: Ver lista de seguidores
# ==========================================
Write-Host "TEST 5: Ver seguidores de User2" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/follows/$global:USER2_ID/followers" `
        -Method GET `
        -Headers @{ "Authorization" = "Bearer $global:TOKEN_USER1" }
    
    Write-Host "[OK] Lista de seguidores obtenida ($($response.Count) seguidores)" -ForegroundColor Green
    foreach ($follower in $response) {
        Write-Host "   - $($follower.username) ($($follower.fullName))" -ForegroundColor Cyan
    }
    Write-Host ""
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 6: Ver lista de seguidos
# ==========================================
Write-Host "TEST 6: Ver seguidos de User1" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/follows/$global:USER1_ID/following" `
        -Method GET `
        -Headers @{ "Authorization" = "Bearer $global:TOKEN_USER1" }
    
    Write-Host "[OK] Lista de seguidos obtenida ($($response.Count) seguidos)" -ForegroundColor Green
    foreach ($following in $response) {
        Write-Host "   - $($following.username) ($($following.fullName))" -ForegroundColor Cyan
    }
    Write-Host ""
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 7: User2 sigue a User1
# ==========================================
Write-Host "TEST 7: User2 sigue a User1" -ForegroundColor Yellow

$followBody2 = @{
    targetUserId = $global:USER1_ID
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/follows" `
        -Method POST `
        -Headers @{ "Authorization" = "Bearer $global:TOKEN_USER2" } `
        -ContentType "application/json" `
        -Body $followBody2
    
    Write-Host "[OK] Follow exitoso (ahora se siguen mutuamente)" -ForegroundColor Green
    Write-Host "   followersCount de User1: $($response.followersCount)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 8: Unfollow - User1 deja de seguir a User2
# ==========================================
Write-Host "TEST 8: User1 deja de seguir a User2" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/follows/$global:USER2_ID" `
        -Method DELETE `
        -Headers @{ "Authorization" = "Bearer $global:TOKEN_USER1" }
    
    Write-Host "[OK] Unfollow exitoso" -ForegroundColor Green
    Write-Host "   isFollowing: $($response.isFollowing)" -ForegroundColor Cyan
    Write-Host "   followersCount: $($response.followersCount)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 9: Intentar unfollow de nuevo (debe fallar)
# ==========================================
Write-Host "TEST 9: Intentar unfollow de nuevo (debe fallar)" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod `
        -Uri "$BASE_URL/auth/follows/$global:USER2_ID" `
        -Method DELETE `
        -Headers @{ "Authorization" = "Bearer $global:TOKEN_USER1" }
    
    Write-Host "[FAIL] No deberia haber funcionado" -ForegroundColor Red
    Write-Host ""
} catch {
    Write-Host "[OK] Error esperado (no estaba siguiendo al usuario)" -ForegroundColor Green
    Write-Host ""
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "PRUEBAS COMPLETADAS" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Resumen:" -ForegroundColor Yellow
Write-Host "- User1 ID: $global:USER1_ID" -ForegroundColor Gray
Write-Host "- User2 ID: $global:USER2_ID" -ForegroundColor Gray
Write-Host "- Tokens obtenidos y funcionales" -ForegroundColor Gray

