# Script para iniciar el Auth Service
Write-Host "Iniciando Auth Service..." -ForegroundColor Cyan

# Ir al directorio correcto
Set-Location C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\auth-service

# Configurar Java 21
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-21'

# Configurar variables de entorno requeridas por Firebase
Write-Host "Configurando variables de entorno..." -ForegroundColor Yellow

# Firebase API Key (Web API Key de tu proyecto Firebase)
# IMPORTANTE: Debe ser el "Web API Key" (sin restricciones)
# Obtenerlo de: https://console.firebase.google.com/project/upsglam-8c88f/settings/general
# NO usar Server Key ni Restricted Key - solo funciona el Web API Key
$env:FIREBASE_API_KEY = "AIzaSyBYcnFxABxm3eyFpCD-nioQbZV1-NDzA5A"

# Opcional: Otras variables si las necesitas
$env:FIREBASE_PROJECT_ID = "upsglam-8c88f"
$env:FIREBASE_STORAGE_BUCKET = "upsglam-8c88f.appspot.com"
$env:FIREBASE_DATABASE_ID = "db-auth"

Write-Host "Variables configuradas:" -ForegroundColor Green
Write-Host "  FIREBASE_PROJECT_ID: $env:FIREBASE_PROJECT_ID" -ForegroundColor Gray
Write-Host "  FIREBASE_DATABASE_ID: $env:FIREBASE_DATABASE_ID" -ForegroundColor Gray
Write-Host "  FIREBASE_API_KEY: $($env:FIREBASE_API_KEY.Substring(0,20))..." -ForegroundColor Gray

# Recompilar
Write-Host "`nCompilando..." -ForegroundColor Yellow
mvn package -DskipTests -q

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilacion exitosa" -ForegroundColor Green
    Write-Host "Iniciando servidor en http://localhost:8082..." -ForegroundColor Cyan
    Write-Host "Presiona Ctrl+C para detener" -ForegroundColor Gray
    Write-Host ""
    
    # Iniciar servidor
    java -jar target\auth-service-1.0.0.jar
} else {
    Write-Host "Error en compilacion" -ForegroundColor Red
}
