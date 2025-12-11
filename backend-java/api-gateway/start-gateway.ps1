# Script para iniciar el API Gateway
Write-Host "Iniciando API Gateway..." -ForegroundColor Cyan

# Ir al directorio correcto
Set-Location $PSScriptRoot

# Configurar Java 21
$env:JAVA_HOME = 'C:\Program Files\jdk-21.0.6+7'

# Recompilar
Write-Host "Compilando..." -ForegroundColor Yellow
mvn package -DskipTests -q

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilacion exitosa" -ForegroundColor Green
    Write-Host "Iniciando API Gateway en http://localhost:8080..." -ForegroundColor Cyan
    Write-Host "Enrutando a:" -ForegroundColor Gray
    Write-Host "  - Auth Service: http://localhost:8082" -ForegroundColor Gray
    Write-Host "  - Post Service: http://localhost:8081" -ForegroundColor Gray
    Write-Host "  - CUDA Service: http://localhost:5000" -ForegroundColor Gray
    Write-Host "Presiona Ctrl+C para detener" -ForegroundColor Gray
    Write-Host ""
    
    # Iniciar servidor
    java -jar target\api-gateway-1.0.0.jar
} else {
    Write-Host "Error en compilacion" -ForegroundColor Red
}
