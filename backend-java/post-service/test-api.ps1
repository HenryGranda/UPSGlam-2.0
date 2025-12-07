# Script para probar el Post Service
Write-Host "🧪 Probando Post Service en http://localhost:8081" -ForegroundColor Cyan
Write-Host ""

# 1. Health Check
Write-Host "1️⃣  Probando Health Check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8081/actuator/health" -Method Get
    Write-Host "✅ Servidor funcionando!" -ForegroundColor Green
    $health | ConvertTo-Json
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}

Write-Host "`n---`n"

# 2. Probar endpoint Feed (fallará sin BD)
Write-Host "2️⃣  Probando GET /feed (fallará sin BD)..." -ForegroundColor Yellow
try {
    $headers = @{
        "X-User-Id" = "test-user-123"
    }
    $feed = Invoke-RestMethod -Uri "http://localhost:8081/feed?page=0&size=10" -Method Get -Headers $headers
    Write-Host "✅ Respuesta recibida!" -ForegroundColor Green
    $feed | ConvertTo-Json
} catch {
    Write-Host "❌ Esperado - Error sin BD: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n---`n"

# 3. Probar crear post (fallará sin BD)
Write-Host "3️⃣  Probando POST /posts (fallará sin BD)..." -ForegroundColor Yellow
try {
    $headers = @{
        "X-User-Id" = "test-user-123"
        "X-Username" = "pepito"
        "Content-Type" = "application/json"
    }
    
    $body = @{
        mediaUrl = "https://example.com/image.jpg"
        filter = "gaussian"
        caption = "Mi primer post"
        mediaType = "image"
    } | ConvertTo-Json
    
    $post = Invoke-RestMethod -Uri "http://localhost:8081/posts" -Method Post -Headers $headers -Body $body
    Write-Host "✅ Post creado!" -ForegroundColor Green
    $post | ConvertTo-Json
} catch {
    Write-Host "❌ Esperado - Error sin BD: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n---`n"
Write-Host "✅ Tests completados!" -ForegroundColor Cyan
Write-Host "💡 Para que funcionen completamente, necesitas configurar la base de datos" -ForegroundColor Yellow
Write-Host "📖 Lee TESTING-GUIDE.md para más info" -ForegroundColor Yellow
