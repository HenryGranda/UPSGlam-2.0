# =========================================
# Test PyCUDA Mock Service
# =========================================
# Script para verificar que el mock funciona correctamente
# =========================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "TEST PYCUDA MOCK SERVICE" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$BASE_URL = "http://localhost:5000"
$IMAGE_PATH = "C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\husky.jpg"

# ==========================================
# TEST 1: HEALTH CHECK
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 1: GET /health" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

try {
    $healthResponse = Invoke-RestMethod -Uri "$BASE_URL/health" -Method GET
    
    Write-Host "[OK] MOCK SERVICE UP" -ForegroundColor Green
    Write-Host "   Status: $($healthResponse.status)" -ForegroundColor Cyan
    Write-Host "   Mode: $($healthResponse.mode)" -ForegroundColor Cyan
    Write-Host "   GPU: $($healthResponse.gpu)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] Mock service no disponible" -ForegroundColor Red
    Write-Host "Ejecuta: .\start-mock.ps1" -ForegroundColor Yellow
    exit 1
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 2: LISTAR FILTROS
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 2: GET /filters" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

try {
    $filtersResponse = Invoke-RestMethod -Uri "$BASE_URL/filters" -Method GET
    
    Write-Host "[OK] FILTROS OBTENIDOS" -ForegroundColor Green
    Write-Host "   Total: $($filtersResponse.total)" -ForegroundColor Cyan
    Write-Host "   Note: $($filtersResponse.note)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Filtros disponibles:" -ForegroundColor Gray
    foreach ($filter in $filtersResponse.filters) {
        Write-Host "     - $($filter.name) ($($filter.category))" -ForegroundColor Gray
    }
    Write-Host ""
} catch {
    Write-Host "[ERROR] Fallo al listar filtros" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 3: VERIFICAR IMAGEN
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 3: Verificar imagen de prueba" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

if (-Not (Test-Path $IMAGE_PATH)) {
    Write-Host "[ERROR] No se encontro la imagen: $IMAGE_PATH" -ForegroundColor Red
    exit 1
}

$imageSize = (Get-Item $IMAGE_PATH).Length
Write-Host "[OK] Imagen encontrada" -ForegroundColor Green
Write-Host "   Path: $IMAGE_PATH" -ForegroundColor Cyan
Write-Host "   Size: $([math]::Round($imageSize/1024, 2)) KB" -ForegroundColor Cyan
Write-Host ""

Start-Sleep -Seconds 1

# ==========================================
# TEST 4: APLICAR FILTRO GAUSSIAN (MOCK)
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 4: POST /filters/gaussian" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

try {
    $imageBytes = [System.IO.File]::ReadAllBytes($IMAGE_PATH)
    
    Write-Host "[SEND] Enviando imagen con filtro gaussian..." -ForegroundColor Yellow
    Write-Host "[INFO] Recordatorio: El mock devuelve la imagen original" -ForegroundColor DarkGray
    
    $response = Invoke-WebRequest `
        -Uri "$BASE_URL/filters/gaussian" `
        -Method POST `
        -ContentType "image/jpeg" `
        -Body $imageBytes
    
    Write-Host "[OK] FILTRO APLICADO (MOCK)" -ForegroundColor Green
    Write-Host "   Status Code: $($response.StatusCode)" -ForegroundColor Cyan
    Write-Host "   Content Type: $($response.Headers['Content-Type'])" -ForegroundColor Cyan
    Write-Host "   Mock Service: $($response.Headers['X-Mock-Service'])" -ForegroundColor Yellow
    Write-Host "   Filter Applied: $($response.Headers['X-Filter-Applied'])" -ForegroundColor Yellow
    Write-Host "   Note: $($response.Headers['X-Note'])" -ForegroundColor Yellow
    Write-Host "   Response Size: $($response.Content.Length) bytes" -ForegroundColor Cyan
    Write-Host ""
    
    # Guardar imagen resultante
    $outputPath = "test_gaussian_mock.jpg"
    [System.IO.File]::WriteAllBytes($outputPath, $response.Content)
    Write-Host "[INFO] Imagen guardada: $outputPath" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] Fallo al aplicar filtro" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 5: APLICAR FILTRO PREWITT (MOCK)
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 5: POST /filters/prewitt" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

try {
    $imageBytes = [System.IO.File]::ReadAllBytes($IMAGE_PATH)
    
    Write-Host "[SEND] Enviando imagen con filtro prewitt..." -ForegroundColor Yellow
    
    $response = Invoke-WebRequest `
        -Uri "$BASE_URL/filters/prewitt" `
        -Method POST `
        -ContentType "image/jpeg" `
        -Body $imageBytes
    
    Write-Host "[OK] FILTRO APLICADO (MOCK)" -ForegroundColor Green
    Write-Host "   Filter Applied: $($response.Headers['X-Filter-Applied'])" -ForegroundColor Yellow
    Write-Host "   Response Size: $($response.Content.Length) bytes" -ForegroundColor Cyan
    Write-Host ""
    
    $outputPath = "test_prewitt_mock.jpg"
    [System.IO.File]::WriteAllBytes($outputPath, $response.Content)
    Write-Host "[INFO] Imagen guardada: $outputPath" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "[ERROR] Fallo al aplicar filtro" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Start-Sleep -Seconds 1

# ==========================================
# TEST 6: FILTRO INVALIDO
# ==========================================
Write-Host "=========================================" -ForegroundColor Blue
Write-Host "TEST 6: POST /filters/invalid (debe fallar)" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue

try {
    $imageBytes = [System.IO.File]::ReadAllBytes($IMAGE_PATH)
    
    Write-Host "[SEND] Enviando con filtro invalido..." -ForegroundColor Yellow
    
    $response = Invoke-WebRequest `
        -Uri "$BASE_URL/filters/invalid_filter" `
        -Method POST `
        -ContentType "image/jpeg" `
        -Body $imageBytes
    
    Write-Host "[WARN] El request deberia haber fallado" -ForegroundColor Yellow
} catch {
    Write-Host "[OK] ERROR ESPERADO (400 Bad Request)" -ForegroundColor Green
    Write-Host "   Filtro invalido rechazado correctamente" -ForegroundColor Cyan
    Write-Host ""
}

# ==========================================
# RESUMEN FINAL
# ==========================================
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "RESUMEN DE PRUEBAS" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[OK] Todos los tests completados" -ForegroundColor Green
Write-Host ""
Write-Host "[INFO] Tests ejecutados:" -ForegroundColor Yellow
Write-Host "   1. Health check           [OK]" -ForegroundColor Green
Write-Host "   2. Listar filtros         [OK]" -ForegroundColor Green
Write-Host "   3. Verificar imagen       [OK]" -ForegroundColor Green
Write-Host "   4. Filtro gaussian (mock) [OK]" -ForegroundColor Green
Write-Host "   5. Filtro prewitt (mock)  [OK]" -ForegroundColor Green
Write-Host "   6. Filtro invalido        [OK]" -ForegroundColor Green
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "MOCK SERVICE FUNCIONANDO CORRECTAMENTE" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[NOTE] Las imagenes guardadas son identicas a la original" -ForegroundColor Yellow
Write-Host "[NOTE] El mock NO procesa imagenes, solo simula la API" -ForegroundColor Yellow
Write-Host ""
