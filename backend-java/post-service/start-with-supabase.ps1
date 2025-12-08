# ========================================
#  POST SERVICE - Supabase Storage Ready
# ========================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  POST SERVICE - Startup Script" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Verificar que el JAR existe
$jarPath = "C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\post-service\target\post-service-1.0.0.jar"
if (-not (Test-Path $jarPath)) {
    Write-Host "❌ ERROR: JAR no encontrado" -ForegroundColor Red
    Write-Host "Ejecuta: mvn clean package -DskipTests" -ForegroundColor Yellow
    exit 1
}

# Mostrar configuración
Write-Host "📦 Configuración:" -ForegroundColor Green
Write-Host "  Puerto: 8081" -ForegroundColor White
Write-Host "  Profile: local" -ForegroundColor White
Write-Host "`n🗄️ Firebase Firestore:" -ForegroundColor Green
Write-Host "  Database: db-auth" -ForegroundColor White
Write-Host "  Collections: posts, likes, comments" -ForegroundColor White
Write-Host "`n☁️ Supabase Storage:" -ForegroundColor Green
Write-Host "  URL: https://opohishcukgkrkfdsgoa.supabase.co" -ForegroundColor White
Write-Host "  Bucket: upsglam (público)" -ForegroundColor White
Write-Host "  Folders: posts/, temp/, avatars/" -ForegroundColor White
Write-Host "`n🎨 Arquitectura Híbrida:" -ForegroundColor Green
Write-Host "  • Metadata → Firestore (posts, likes, comments)" -ForegroundColor White
Write-Host "  • Imágenes → Supabase Storage (CDN público)" -ForegroundColor White
Write-Host "  • Filtros → PyCUDA Service (procesamiento GPU)" -ForegroundColor White

Write-Host "`n📋 Endpoints disponibles:" -ForegroundColor Yellow
Write-Host "  POST   /api/images/upload        - Subir imagen a Supabase" -ForegroundColor Cyan
Write-Host "  POST   /api/images/preview       - Aplicar filtro (PyCUDA)" -ForegroundColor Cyan
Write-Host "  POST   /api/posts                - Crear post" -ForegroundColor Cyan
Write-Host "  GET    /api/feed                 - Obtener feed" -ForegroundColor Cyan
Write-Host "  POST   /api/posts/{id}/likes     - Like post" -ForegroundColor Cyan
Write-Host "  POST   /api/posts/{id}/comments  - Comentar post" -ForegroundColor Cyan

Write-Host "`n🚀 Iniciando servicio...`n" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

cd C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\post-service
java -jar target\post-service-1.0.0.jar --spring.profiles.active=local
