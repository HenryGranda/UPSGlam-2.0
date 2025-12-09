# Script para iniciar el servidor
Write-Host "Iniciando Post Service..." -ForegroundColor Cyan

# Ir al directorio correcto
Set-Location C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\post-service

# Configurar Java 21
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-21'

# Recompilar
Write-Host "Compilando..." -ForegroundColor Yellow
mvn package -DskipTests -q

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilacion exitosa" -ForegroundColor Green
    Write-Host "Iniciando servidor en http://localhost:8081..." -ForegroundColor Cyan
    Write-Host "Presiona Ctrl+C para detener" -ForegroundColor Gray
    Write-Host ""
    
    # Iniciar servidor
    java -jar target\post-service-1.0.0.jar
} else {
    Write-Host "Error en compilacion" -ForegroundColor Red
}
