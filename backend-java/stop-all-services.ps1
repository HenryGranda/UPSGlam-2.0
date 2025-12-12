# ============================================================
# Stop All UPSGlam Services
# ============================================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "STOPPING ALL UPSGLAM SERVICES" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

docker-compose stop

Write-Host ""
Write-Host "[OK] All services stopped" -ForegroundColor Green
Write-Host ""
Write-Host "To start again: .\start-all-services.ps1" -ForegroundColor Yellow
Write-Host "To remove containers: docker-compose down" -ForegroundColor Yellow
Write-Host ""
