# Script para generar APK de UPSGlam

Write-Host 'Generando APK de UPSGlam...' -ForegroundColor Cyan

# Limpiar build anterior
Write-Host "`nLimpiando builds anteriores..." -ForegroundColor Yellow
flutter clean

# Obtener dependencias
Write-Host "`nObteniendo dependencias..." -ForegroundColor Yellow
flutter pub get

# Generar APK en modo release
Write-Host "`nGenerando APK (esto puede tomar varios minutos)..." -ForegroundColor Yellow
flutter build apk --release

# Verificar que se generó correctamente
$apkPath = 'build\app\outputs\flutter-apk\app-release.apk'

if (Test-Path $apkPath) {
    Write-Host "`nAPK generado exitosamente!" -ForegroundColor Green
    Write-Host "`nUbicacion: $apkPath" -ForegroundColor Cyan

    # Obtener tamaño del APK
    $apkSize = (Get-Item $apkPath).Length / 1MB
    Write-Host ("Tamano: {0} MB" -f ([math]::Round($apkSize, 2))) -ForegroundColor Yellow

    # Abrir carpeta del APK
    Write-Host "`nAbriendo carpeta del APK..." -ForegroundColor Yellow
    explorer 'build\app\outputs\flutter-apk'
}
else {
    Write-Host "`nError: No se pudo generar el APK" -ForegroundColor Red
    Write-Host 'Revisa los errores anteriores' -ForegroundColor Red
}
