# Script para detener, recompilar y reiniciar auth-service

Write-Host "Deteniendo procesos de auth-service..." -ForegroundColor Yellow

# Intentar detener procesos Java que tengan auth-service
Get-Process -Name java -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*auth-service*" -or 
    $_.CommandLine -like "*auth-service*" -or
    $_.MainWindowTitle -like "*auth-service*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

# Intentar encontrar proceso en puerto 8082
$process = Get-NetTCPConnection -LocalPort 8082 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($process) {
    Write-Host "Deteniendo proceso en puerto 8082 (PID: $process)..." -ForegroundColor Yellow
    Stop-Process -Id $process -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2

Write-Host "Recompilando..." -ForegroundColor Cyan
mvn clean package -DskipTests

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Compilacion exitosa!" -ForegroundColor Green
    Write-Host "Iniciando auth-service..." -ForegroundColor Cyan
    Write-Host ""
    
    # Iniciar el servicio
    .\start-auth.ps1
} else {
    Write-Host "Error en la compilacion" -ForegroundColor Red
}
