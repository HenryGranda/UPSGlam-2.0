# Script para iniciar el Post Service
Write-Host "Iniciando Post Service..." -ForegroundColor Cyan

# Ir al directorio correcto
Set-Location C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\post-service

# Configurar Java 21
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-21'

# Configurar variables de entorno de Firebase
Write-Host "Configurando Firebase..." -ForegroundColor Yellow
$env:FIREBASE_API_KEY = "AIzaSyBYcnFxABxm3eyFpCD-nioQbZV1-NDzA5A"
$env:FIREBASE_PROJECT_ID = "upsglam-8c88f"
$env:FIREBASE_STORAGE_BUCKET = "upsglam-8c88f.appspot.com"

# Configurar variables de entorno de Supabase
Write-Host "Configurando Supabase..." -ForegroundColor Yellow
$env:SUPABASE_URL = "https://opohishcukgkrkfdsgoa.supabase.co"
$env:SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9wb2hpc2hjdWtna3JrZmRzZ29hIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM2ODU0NjQsImV4cCI6MjA0OTI2MTQ2NH0.tW8eYXEUW7e26zRH6r7pVHaZEX2fq2SkYbZg8rblKl8"
$env:SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9wb2hpc2hjdWtna3JrZmRzZ29hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczMzY4NTQ2NCwiZXhwIjoyMDQ5MjYxNDY0fQ.fZ-9hPTJf8MJKW2sSNXNVL8tnJgRDxP_JGmQm6Nh_H0"

Write-Host "Variables configuradas:" -ForegroundColor Green
Write-Host "  FIREBASE_PROJECT_ID: $env:FIREBASE_PROJECT_ID" -ForegroundColor Gray
Write-Host "  SUPABASE_URL: $env:SUPABASE_URL" -ForegroundColor Gray

# Recompilar
Write-Host "`nCompilando..." -ForegroundColor Yellow
mvn package -DskipTests -q

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilacion exitosa" -ForegroundColor Green
    Write-Host "Iniciando servidor en http://localhost:8081..." -ForegroundColor Cyan
    Write-Host "Presiona Ctrl+C para detener`n" -ForegroundColor Yellow
    
    # Ejecutar el JAR
    java -jar target\post-service-1.0.0.jar --spring.profiles.active=local
} else {
    Write-Host "Error en compilacion" -ForegroundColor Red
    exit 1
}
