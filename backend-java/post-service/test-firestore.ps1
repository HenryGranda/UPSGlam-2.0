# Script de prueba para post-service con Firestore
# UPSGlam 2.0 - Test endpoints

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Test Post Service con Firestore" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$baseUrl = "http://localhost:8081"

# 1. Health Check
Write-Host "1. Probando Health Check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/api/health" -Method GET
    Write-Host "✓ Health: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "✗ Error en health check" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}
Write-Host ""

# 2. Crear un post de prueba
Write-Host "2. Creando post de prueba en Firestore..." -ForegroundColor Yellow
$postData = @{
    userId = "testUser123"
    username = "testuser"
    userPhotoUrl = "https://example.com/photo.jpg"
    imageUrl = "https://example.com/post.jpg"
    filter = "ups_logo"
    description = "Post de prueba en Firestore desde PowerShell"
} | ConvertTo-Json

try {
    $newPost = Invoke-RestMethod -Uri "$baseUrl/api/posts" -Method POST -Body $postData -ContentType "application/json"
    Write-Host "✓ Post creado: $($newPost.id)" -ForegroundColor Green
    Write-Host "  Description: $($newPost.description)" -ForegroundColor Gray
    Write-Host "  Likes: $($newPost.likesCount)" -ForegroundColor Gray
    Write-Host "  Comments: $($newPost.commentsCount)" -ForegroundColor Gray
    
    $postId = $newPost.id
} catch {
    Write-Host "✗ Error creando post" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    $postId = $null
}
Write-Host ""

# 3. Obtener el feed
Write-Host "3. Obteniendo feed (posts recientes)..." -ForegroundColor Yellow
try {
    $feed = Invoke-RestMethod -Uri "$baseUrl/api/feed?page=0&size=5" -Method GET
    Write-Host "✓ Feed obtenido: $($feed.totalItems) posts" -ForegroundColor Green
    Write-Host "  Página: $($feed.page)" -ForegroundColor Gray
    Write-Host "  Size: $($feed.size)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Error obteniendo feed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}
Write-Host ""

# 4. Obtener detalle de un post (si se creó)
if ($postId) {
    Write-Host "4. Obteniendo detalle del post..." -ForegroundColor Yellow
    try {
        $postDetail = Invoke-RestMethod -Uri "$baseUrl/api/posts/$postId" -Method GET
        Write-Host "✓ Post obtenido: $($postDetail.id)" -ForegroundColor Green
        Write-Host "  Username: $($postDetail.username)" -ForegroundColor Gray
        Write-Host "  Description: $($postDetail.description)" -ForegroundColor Gray
    } catch {
        Write-Host "✗ Error obteniendo post" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
    Write-Host ""
    
    # 5. Agregar un like
    Write-Host "5. Agregando like al post..." -ForegroundColor Yellow
    try {
        $likeResult = Invoke-RestMethod -Uri "$baseUrl/api/posts/$postId/likes" -Method POST
        Write-Host "✓ Like agregado" -ForegroundColor Green
        Write-Host "  Post ID: $($likeResult.postId)" -ForegroundColor Gray
        Write-Host "  Likes Count: $($likeResult.likesCount)" -ForegroundColor Gray
        Write-Host "  Liked by me: $($likeResult.likedByMe)" -ForegroundColor Gray
    } catch {
        Write-Host "✗ Error agregando like" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
    Write-Host ""
    
    # 6. Agregar un comentario
    Write-Host "6. Agregando comentario al post..." -ForegroundColor Yellow
    $commentData = @{
        text = "Este es un comentario de prueba desde PowerShell!"
    } | ConvertTo-Json
    
    try {
        $comment = Invoke-RestMethod -Uri "$baseUrl/api/posts/$postId/comments" -Method POST -Body $commentData -ContentType "application/json"
        Write-Host "✓ Comentario agregado: $($comment.id)" -ForegroundColor Green
        Write-Host "  Username: $($comment.username)" -ForegroundColor Gray
        Write-Host "  Text: $($comment.text)" -ForegroundColor Gray
    } catch {
        Write-Host "✗ Error agregando comentario" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
    Write-Host ""
    
    # 7. Obtener comentarios del post
    Write-Host "7. Obteniendo comentarios del post..." -ForegroundColor Yellow
    try {
        $comments = Invoke-RestMethod -Uri "$baseUrl/api/posts/$postId/comments?page=0&size=10" -Method GET
        Write-Host "✓ Comentarios obtenidos: $($comments.totalItems) comentarios" -ForegroundColor Green
    } catch {
        Write-Host "✗ Error obteniendo comentarios" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Tests completados" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
