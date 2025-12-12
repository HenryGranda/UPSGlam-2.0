# Script para instalar el APK en dispositivo Android conectado
# Uso: .\install-apk.ps1

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  UPSGlam 2.0 - Instalador de APK" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

$apkPath = ".\build\app\outputs\flutter-apk\app-release.apk"

# Verificar que existe el APK
if (-Not (Test-Path $apkPath)) {
    Write-Host "ERROR: No se encontro el APK en: $apkPath" -ForegroundColor Red
    Write-Host "Por favor ejecuta primero: flutter build apk --release" -ForegroundColor Yellow
    exit 1
}

# Obtener info del APK
$apkInfo = Get-Item $apkPath
$apkSizeMB = [math]::Round($apkInfo.Length / 1MB, 2)

Write-Host "APK encontrado:" -ForegroundColor Green
Write-Host "  Ruta: $apkPath" -ForegroundColor White
Write-Host "  Tamano: $apkSizeMB MB" -ForegroundColor White
Write-Host "  Fecha: $($apkInfo.LastWriteTime)" -ForegroundColor White
Write-Host ""

# Verificar si hay dispositivo conectado
Write-Host "Verificando dispositivos conectados..." -ForegroundColor Yellow
$devices = adb devices

if ($devices -match "device$") {
    Write-Host "Dispositivo Android detectado!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Instalando APK..." -ForegroundColor Yellow
    
    adb install -r $apkPath
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "APK instalado exitosamente!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Iniciando aplicacion..." -ForegroundColor Yellow
        adb shell am start -n ec.edu.ups.upsglam/.MainActivity
        Write-Host ""
        Write-Host "Listo! La aplicacion deberia estar ejecutandose en tu dispositivo." -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "ERROR: Fallo la instalacion del APK" -ForegroundColor Red
        Write-Host "Verifica que el dispositivo tenga habilitada la depuracion USB" -ForegroundColor Yellow
    }
} else {
    Write-Host "No se detecto ningun dispositivo Android conectado" -ForegroundColor Red
    Write-Host ""
    Write-Host "Opciones:" -ForegroundColor Yellow
    Write-Host "  1. Conecta tu dispositivo Android por USB" -ForegroundColor White
    Write-Host "  2. Habilita la depuracion USB en Ajustes > Opciones de desarrollador" -ForegroundColor White
    Write-Host "  3. Ejecuta 'adb devices' para verificar la conexion" -ForegroundColor White
    Write-Host ""
    Write-Host "O puedes copiar manualmente el APK desde:" -ForegroundColor Cyan
    Write-Host "  $apkPath" -ForegroundColor White
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
