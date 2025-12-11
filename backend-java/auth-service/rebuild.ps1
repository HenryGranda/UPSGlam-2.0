# Script para recompilar y reiniciar auth-service
Write-Host "Recompilando auth-service..." -ForegroundColor Yellow

# Compilar
mvn clean package -DskipTests

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilacion exitosa!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Ahora ejecuta en la terminal de Java:" -ForegroundColor Cyan
    Write-Host ".\start-auth.ps1" -ForegroundColor White
} else {
    Write-Host "Error en la compilacion" -ForegroundColor Red
}
