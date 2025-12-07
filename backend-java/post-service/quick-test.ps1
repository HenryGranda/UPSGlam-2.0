Write-Host "🧪 Probando endpoints del Post Service" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Probando GET /feed..." -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "http://localhost:8081/feed" -Method Get -Headers @{"X-User-Id"="test-123"}
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host "`n2. Probando GET /actuator/health..." -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "http://localhost:8081/actuator/health" -Method Get
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
