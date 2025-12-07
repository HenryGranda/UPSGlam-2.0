# Script simple para probar endpoints que funcionan sin base de datos
$baseUrl = "http://localhost:8081/api"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  PROBANDO ENDPOINTS - UPSGlam 2.0" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. Health Check
Write-Host "1. Health Check" -ForegroundColor Yellow
Write-Host "   GET $baseUrl/health" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "   Status: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.status)" -ForegroundColor Green
    Write-Host "   Service: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.service)" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host ""
}

# 2. Root endpoint
Write-Host "2. Root Endpoint" -ForegroundColor Yellow
Write-Host "   GET $baseUrl/" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/" -Method Get
    Write-Host "   Message: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.message)" -ForegroundColor Green
    Write-Host "   Version: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.version)" -ForegroundColor Green
    Write-Host "   Endpoints: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.endpoints)" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host ""
}

# 3. Actuator Health
Write-Host "3. Actuator Health" -ForegroundColor Yellow
Write-Host "   GET http://localhost:8081/actuator/health" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8081/actuator/health" -Method Get
    Write-Host "   Status: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.status)" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PRUEBAS COMPLETADAS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Nota: Los endpoints de posts/likes/comentarios requieren" -ForegroundColor Yellow
Write-Host "      conexion a Supabase (configurar credenciales en .env)`n" -ForegroundColor Yellow
