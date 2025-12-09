# Script para probar Auth Service
$baseUrl = "http://localhost:8082/api"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  PROBANDO AUTH SERVICE - UPSGlam 2.0" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. Health Check
Write-Host "1. Health Check" -ForegroundColor Yellow
Write-Host "   GET $baseUrl/../health" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8082/health" -Method Get
    Write-Host "   Status: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.status)" -ForegroundColor Green
    Write-Host "   Service: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.service)" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host ""
}

# 2. Root Info
Write-Host "2. Service Info" -ForegroundColor Yellow
Write-Host "   GET http://localhost:8082/" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8082/" -Method Get
    Write-Host "   Service: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.service)" -ForegroundColor Green
    Write-Host "   Version: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.version)" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ENDPOINTS DISPONIBLES" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "POST $baseUrl/auth/register" -ForegroundColor White
Write-Host "POST $baseUrl/auth/login" -ForegroundColor White
Write-Host "GET  $baseUrl/auth/me" -ForegroundColor White
Write-Host "PATCH $baseUrl/users/me`n" -ForegroundColor White

Write-Host "Nota: Los endpoints de autenticacion requieren" -ForegroundColor Yellow
Write-Host "      credenciales de Firebase configuradas.`n" -ForegroundColor Yellow
