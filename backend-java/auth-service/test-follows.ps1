# Script de prueba para API de Follows
# Asegurate de tener API Gateway (8080) y auth-service (8082) corriendo

$BASE_URL = "http://localhost:8082/api"  # API Gateway

# IMPORTANTE: Reemplaza estos valores con tokens reales de Firebase
$TOKEN_USER1 = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImY..."  # Token del usuario 1
$TOKEN_USER2 = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImY..."  # Token del usuario 2
$USER2_ID = "uid_del_usuario_2"  # UID del usuario 2

Write-Host "=== PRUEBAS DE API DE FOLLOWS ===" -ForegroundColor Cyan
Write-Host ""

# Test 1: User1 sigue a User2
Write-Host "Test 1: User1 sigue a User2" -ForegroundColor Yellow
$followBody = @{
    targetUserId = $USER2_ID
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/follows" `
        -Method POST `
        -Headers @{
            "Authorization" = "Bearer $TOKEN_USER1"
            "Content-Type" = "application/json"
        } `
        -Body $followBody
    
    Write-Host "[OK] Follow exitoso" -ForegroundColor Green
    Write-Host $($response | ConvertTo-Json -Depth 10)
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 1

# Test 2: Intentar seguir de nuevo (debe fallar con ALREADY_FOLLOWING)
Write-Host "Test 2: Intentar seguir de nuevo (debe fallar)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/follows" `
        -Method POST `
        -Headers @{
            "Authorization" = "Bearer $TOKEN_USER1"
            "Content-Type" = "application/json"
        } `
        -Body $followBody
    
    Write-Host "[FAIL] No deberia haber funcionado" -ForegroundColor Red
} catch {
    Write-Host "[OK] Error esperado" -ForegroundColor Green
}

Write-Host ""
Start-Sleep -Seconds 1

# Test 3: Ver estadisticas de User2
Write-Host "Test 3: Ver estadisticas de User2" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/follows/$USER2_ID/stats" `
        -Method GET `
        -Headers @{
            "Authorization" = "Bearer $TOKEN_USER1"
        }
    
    Write-Host "[OK] Estadisticas obtenidas" -ForegroundColor Green
    Write-Host $($response | ConvertTo-Json -Depth 10)
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 1

# Test 4: Ver estadisticas con listas
Write-Host "Test 4: Ver estadisticas con listas completas" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/follows/$USER2_ID/stats?includeList=true" `
        -Method GET `
        -Headers @{
            "Authorization" = "Bearer $TOKEN_USER1"
        }
    
    Write-Host "[OK] Estadisticas con listas obtenidas" -ForegroundColor Green
    Write-Host $($response | ConvertTo-Json -Depth 10)
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 1

# Test 5: Ver lista de seguidores de User2
Write-Host "Test 5: Ver seguidores de User2" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/follows/$USER2_ID/followers" `
        -Method GET `
        -Headers @{
            "Authorization" = "Bearer $TOKEN_USER1"
        }
    
    Write-Host "[OK] Lista de seguidores obtenida" -ForegroundColor Green
    Write-Host $($response | ConvertTo-Json -Depth 10)
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 1

# Test 6: Ver lista de seguidos de User1
Write-Host "Test 6: Ver seguidos de User1" -ForegroundColor Yellow
try {
    $currentUserId = "uid_del_usuario_1"  # Reemplazar con UID de user1
    $response = Invoke-RestMethod -Uri "$BASE_URL/follows/$currentUserId/following" `
        -Method GET `
        -Headers @{
            "Authorization" = "Bearer $TOKEN_USER2"
        }
    
    Write-Host "[OK] Lista de seguidos obtenida" -ForegroundColor Green
    Write-Host $($response | ConvertTo-Json -Depth 10)
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 1

# Test 7: Unfollow
Write-Host "Test 7: User1 deja de seguir a User2" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/follows/$USER2_ID" `
        -Method DELETE `
        -Headers @{
            "Authorization" = "Bearer $TOKEN_USER1"
        }
    
    Write-Host "[OK] Unfollow exitoso" -ForegroundColor Green
    Write-Host $($response | ConvertTo-Json -Depth 10)
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 1

# Test 8: Intentar unfollow de nuevo (debe fallar)
Write-Host "Test 8: Intentar unfollow de nuevo (debe fallar)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/follows/$USER2_ID" `
        -Method DELETE `
        -Headers @{
            "Authorization" = "Bearer $TOKEN_USER1"
        }
    
    Write-Host "[FAIL] No deberia haber funcionado" -ForegroundColor Red
} catch {
    Write-Host "[OK] Error esperado" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== PRUEBAS COMPLETADAS ===" -ForegroundColor Cyan

# Notas de uso:
Write-Host ""
Write-Host "NOTA: Para usar este script necesitas:" -ForegroundColor Yellow
Write-Host "1. Tener API Gateway corriendo en puerto 8080" -ForegroundColor White
Write-Host "2. Tener auth-service corriendo en puerto 8082" -ForegroundColor White
Write-Host "3. Obtener tokens reales de Firebase Auth" -ForegroundColor White
Write-Host "4. Reemplazar las variables TOKEN_USER1, TOKEN_USER2, USER2_ID" -ForegroundColor White
Write-Host ""
Write-Host "Para obtener un token:" -ForegroundColor Yellow
Write-Host "- Login desde tu app movil o frontend" -ForegroundColor White
Write-Host "- Copiar el idToken del response" -ForegroundColor White
