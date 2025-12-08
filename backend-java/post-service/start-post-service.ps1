# Script para compilar y ejecutar el post-service en puerto 8081
# UPSGlam 2.0 - Post Service con Firestore

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  UPSGlam Post Service con Firestore" -ForegroundColor Cyan
Write-Host "  Puerto: 8081" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que existe el archivo de credenciales
$credentialsPath = "src/main/resources/firebase-credentials.json"
if (-Not (Test-Path $credentialsPath)) {
    Write-Host "ERROR: No se encontró firebase-credentials.json" -ForegroundColor Red
    Write-Host "Ejecuta: Copy-Item desde auth-service" -ForegroundColor Yellow
    exit 1
}

$localConfigPath = "src/main/resources/application-local.yml"
if (-Not (Test-Path $localConfigPath)) {
    Write-Host "ERROR: No se encontró application-local.yml" -ForegroundColor Red
    Write-Host "Ejecuta: Copy-Item desde auth-service" -ForegroundColor Yellow
    exit 1
}

Write-Host "Credenciales Firebase encontradas ✓" -ForegroundColor Green
Write-Host ""

# Compilar el proyecto
Write-Host "Compilando post-service..." -ForegroundColor Yellow
mvn clean package -DskipTests

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Falló la compilación" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Compilación exitosa ✓" -ForegroundColor Green
Write-Host ""

# Ejecutar el servicio
Write-Host "Iniciando post-service en puerto 8081..." -ForegroundColor Yellow
Write-Host "Usando perfil: local (con Firebase API Key)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Presiona Ctrl+C para detener el servicio" -ForegroundColor Gray
Write-Host ""

java "-Dspring.profiles.active=local" -jar target/post-service-1.0.0.jar
