# Script para probar todos los endpoints del Post Service
# Asegúrate de que el servidor esté corriendo en http://localhost:8081

$baseUrl = "http://localhost:8081/api"
$headers = @{
    "X-User-Id" = "user-123"
    "X-Username" = "testuser"
    "Content-Type" = "application/json"
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  PROBANDO POST SERVICE - UPSGlam 2.0" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. Health Check
Write-Host "1. Health Check" -ForegroundColor Yellow
Write-Host "   GET $baseUrl/health" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "   ✓ Status: $($response.status)" -ForegroundColor Green
    Write-Host "   ✓ Service: $($response.service)`n" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 2. Get Feed (obtener posts de prueba)
Write-Host "2. Obtener Feed" -ForegroundColor Yellow
Write-Host "   GET $baseUrl/feed" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/feed" -Method Get -Headers $headers
    Write-Host "   ✓ Posts encontrados: $($response.posts.Count)" -ForegroundColor Green
    Write-Host "   ✓ Tiene más: $($response.hasMore)`n" -ForegroundColor Green
    
    # Mostrar detalles de los posts
    foreach ($post in $response.posts) {
        Write-Host "   📷 Post ID: $($post.id)" -ForegroundColor Cyan
        Write-Host "      Usuario: $($post.userId)" -ForegroundColor Gray
        Write-Host "      Descripción: $($post.description)" -ForegroundColor Gray
        Write-Host "      Likes: $($post.likesCount) | Comentarios: $($post.commentsCount)" -ForegroundColor Gray
        Write-Host "      Filtro: $($post.filter)" -ForegroundColor Gray
        Write-Host ""
    }
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 3. Get Post by ID
Write-Host "3. Obtener Post Específico" -ForegroundColor Yellow
Write-Host "   GET $baseUrl/posts/post-1" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/posts/post-1" -Method Get -Headers $headers
    Write-Host "   ✓ Post ID: $($response.id)" -ForegroundColor Green
    Write-Host "   ✓ Descripción: $($response.description)" -ForegroundColor Green
    Write-Host "   ✓ Likes: $($response.likesCount)`n" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 4. Get Posts by User
Write-Host "4. Obtener Posts de Usuario" -ForegroundColor Yellow
Write-Host "   GET $baseUrl/posts/user/user-123" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/posts/user/user-123" -Method Get -Headers $headers
    Write-Host "   ✓ Posts del usuario user-123: $($response.Count)" -ForegroundColor Green
    foreach ($post in $response) {
        Write-Host "      - $($post.id): $($post.description)" -ForegroundColor Gray
    }
    Write-Host ""
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 5. Create New Post
Write-Host "5. Crear Nuevo Post" -ForegroundColor Yellow
Write-Host "   POST $baseUrl/posts" -ForegroundColor Gray
$newPost = @{
    mediaUrl = "https://picsum.photos/400/600?random=999"
    filter = "VINTAGE"
    caption = "Nuevo post de prueba desde PowerShell! #test #api"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/posts" -Method Post -Headers $headers -Body $newPost
    Write-Host "   ✓ Post creado exitosamente!" -ForegroundColor Green
    Write-Host "   ✓ ID: $($response.id)" -ForegroundColor Green
    Write-Host "   ✓ Descripción: $($response.description)`n" -ForegroundColor Green
    $createdPostId = $response.id
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 6. Get Likes for Post
Write-Host "6. Obtener Likes de un Post" -ForegroundColor Yellow
Write-Host "   GET $baseUrl/posts/post-1/likes" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/posts/post-1/likes" -Method Get -Headers $headers
    Write-Host "   ✓ Total de likes: $($response.total)" -ForegroundColor Green
    Write-Host "   ✓ Usuario actual dio like: $($response.likedByCurrentUser)" -ForegroundColor Green
    Write-Host "   ✓ Usuarios que dieron like:" -ForegroundColor Green
    foreach ($like in $response.likes) {
        Write-Host "      - $($like.userId) ($(Get-Date $like.createdAt -Format 'HH:mm'))" -ForegroundColor Gray
    }
    Write-Host ""
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 7. Add Like to Post
Write-Host "7. Dar Like a un Post" -ForegroundColor Yellow
Write-Host "   POST $baseUrl/posts/post-2/likes" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/posts/post-2/likes" -Method Post -Headers $headers
    Write-Host "   ✓ Like agregado exitosamente!" -ForegroundColor Green
    Write-Host "   ✓ Post ID: $($response.postId)" -ForegroundColor Green
    Write-Host "   ✓ Usuario: $($response.userId)`n" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 8. Get Comments for Post
Write-Host "8. Obtener Comentarios de un Post" -ForegroundColor Yellow
Write-Host "   GET $baseUrl/posts/post-1/comments" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/posts/post-1/comments" -Method Get -Headers $headers
    Write-Host "   ✓ Total de comentarios: $($response.Count)" -ForegroundColor Green
    foreach ($comment in $response) {
        Write-Host "   💬 @$($comment.username): $($comment.text)" -ForegroundColor Cyan
        Write-Host "      $(Get-Date $comment.createdAt -Format 'dd/MM/yyyy HH:mm')" -ForegroundColor Gray
    }
    Write-Host ""
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 9. Add Comment to Post
Write-Host "9. Agregar Comentario a un Post" -ForegroundColor Yellow
Write-Host "   POST $baseUrl/posts/post-1/comments" -ForegroundColor Gray
$newComment = @{
    text = "Este es un comentario de prueba desde PowerShell! 🚀"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/posts/post-1/comments" -Method Post -Headers $headers -Body $newComment
    Write-Host "   ✓ Comentario agregado exitosamente!" -ForegroundColor Green
    Write-Host "   ✓ ID: $($response.id)" -ForegroundColor Green
    Write-Host "   ✓ Texto: $($response.text)`n" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 10. Get Comments by User
Write-Host "10. Obtener Comentarios de un Usuario" -ForegroundColor Yellow
Write-Host "    GET $baseUrl/users/user-456/comments" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/users/user-456/comments" -Method Get -Headers $headers
    Write-Host "   ✓ Comentarios del usuario: $($response.Count)" -ForegroundColor Green
    foreach ($comment in $response) {
        Write-Host "      - En post $($comment.postId): $($comment.text)" -ForegroundColor Gray
    }
    Write-Host ""
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 11. Update Post Caption
Write-Host "11. Actualizar Caption de un Post" -ForegroundColor Yellow
Write-Host "    PATCH $baseUrl/posts/post-1/caption" -ForegroundColor Gray
$updateCaption = @{
    caption = "Caption actualizado desde PowerShell! 🎉 #updated"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/posts/post-1/caption" -Method Patch -Headers $headers -Body $updateCaption
    Write-Host "   ✓ Caption actualizado exitosamente!" -ForegroundColor Green
    Write-Host "   ✓ Nueva descripción: $($response.description)`n" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# 12. Remove Like from Post
Write-Host "12. Quitar Like de un Post" -ForegroundColor Yellow
Write-Host "    DELETE $baseUrl/posts/post-2/likes" -ForegroundColor Gray
try {
    Invoke-RestMethod -Uri "$baseUrl/posts/post-2/likes" -Method Delete -Headers $headers
    Write-Host "   ✓ Like eliminado exitosamente!`n" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Error: $_`n" -ForegroundColor Red
}

# Resumen final
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  PRUEBAS COMPLETADAS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Endpoints probados:" -ForegroundColor White
Write-Host "  ✓ Health Check" -ForegroundColor Green
Write-Host "  ✓ Feed de posts" -ForegroundColor Green
Write-Host "  ✓ Obtener post por ID" -ForegroundColor Green
Write-Host "  ✓ Posts por usuario" -ForegroundColor Green
Write-Host "  ✓ Crear post" -ForegroundColor Green
Write-Host "  ✓ Obtener likes" -ForegroundColor Green
Write-Host "  ✓ Agregar like" -ForegroundColor Green
Write-Host "  ✓ Quitar like" -ForegroundColor Green
Write-Host "  ✓ Obtener comentarios" -ForegroundColor Green
Write-Host "  ✓ Agregar comentario" -ForegroundColor Green
Write-Host "  ✓ Comentarios por usuario" -ForegroundColor Green
Write-Host "  ✓ Actualizar caption`n" -ForegroundColor Green

Write-Host "Servidor: $baseUrl" -ForegroundColor Gray
Write-Host "Usuario de prueba: user-123 (testuser)`n" -ForegroundColor Gray
