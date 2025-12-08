# Script para probar el API Gateway
$gateway = "http://localhost:8080"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  PROBANDO API GATEWAY - UPSGlam 2.0" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. Gateway Info
Write-Host "1. Gateway Info" -ForegroundColor Yellow
Write-Host "   GET $gateway/" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$gateway/" -Method Get
    Write-Host "   Service: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.service)" -ForegroundColor Green
    Write-Host "   Version: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.version)" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host ""
}

# 2. Gateway Health
Write-Host "2. Gateway Health" -ForegroundColor Yellow
Write-Host "   GET $gateway/health" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$gateway/health" -Method Get
    Write-Host "   Status: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.status)" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host ""
}

# 3. Post Service via Gateway
Write-Host "3. Post Service via Gateway" -ForegroundColor Yellow
Write-Host "   GET $gateway/api/health/posts" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$gateway/api/health/posts" -Method Get
    Write-Host "   Status: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.status)" -ForegroundColor Green
    Write-Host "   Service: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.service)" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host ""
}

# 4. Feed via Gateway
Write-Host "4. Feed via Gateway" -ForegroundColor Yellow
Write-Host "   GET $gateway/api/feed" -ForegroundColor Gray
$headers = @{
    "X-User-Id" = "user-123"
    "X-Username" = "testuser"
}
try {
    $response = Invoke-RestMethod -Uri "$gateway/api/feed" -Method Get -Headers $headers
    Write-Host "   Posts: " -NoNewline -ForegroundColor Gray
    Write-Host "$($response.posts.Count)" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

# 5. Actuator Health
Write-Host "5. Gateway Actuator Health" -ForegroundColor Yellow
Write-Host "   GET $gateway/actuator/health" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$gateway/actuator/health" -Method Get
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

Write-Host "API Gateway: http://localhost:8080" -ForegroundColor White
Write-Host "Post Service: http://localhost:8081" -ForegroundColor Gray
Write-Host "CUDA Service: http://localhost:5000" -ForegroundColor Gray
Write-Host ""
