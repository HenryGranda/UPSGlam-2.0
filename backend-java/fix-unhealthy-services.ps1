# Script para reiniciar servicios problematicos

Write-Host "Revisando servicios unhealthy..." -ForegroundColor Cyan
Write-Host ""

# Ver logs de auth-service
Write-Host "Logs de auth-service (ultimas 20 lineas):" -ForegroundColor Yellow
docker logs --tail 20 upsglam-auth-service
Write-Host ""

# Ver logs de post-service
Write-Host "Logs de post-service (ultimas 20 lineas):" -ForegroundColor Yellow
docker logs --tail 20 upsglam-post-service
Write-Host ""

# Reiniciar servicios
Write-Host "Reiniciando servicios..." -ForegroundColor Yellow
docker-compose restart auth-service post-service

Write-Host ""
Write-Host "Esperando 30 segundos para que los servicios inicien..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

Write-Host ""
Write-Host "Estado de los contenedores:" -ForegroundColor Cyan
docker ps

Write-Host ""
Write-Host "Reinicio completado. Verifica si ahora estan healthy." -ForegroundColor Green
