# ============================================================
# Build and Run CUDA Lab Backend Docker Container
# ============================================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "CUDA LAB BACKEND - DOCKER BUILD" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "[1/6] Verificando Docker..." -ForegroundColor Yellow
try {
    docker info > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Docker no esta corriendo. Inicia Docker Desktop." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Docker esta corriendo" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Docker no esta instalado o no esta corriendo" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Check NVIDIA GPU availability
Write-Host "[2/6] Verificando GPU NVIDIA..." -ForegroundColor Yellow
try {
    $gpuCheck = docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] GPU NVIDIA detectada y accesible en Docker" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] GPU no detectada. El container podria no funcionar correctamente." -ForegroundColor Yellow
        Write-Host "         Asegurate de tener NVIDIA Container Toolkit instalado." -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] No se pudo verificar GPU" -ForegroundColor Yellow
}
Write-Host ""

# Stop and remove existing container
Write-Host "[3/6] Limpiando containers existentes..." -ForegroundColor Yellow
docker stop cuda-lab-backend 2>$null
docker rm cuda-lab-backend 2>$null
Write-Host "[OK] Limpieza completada" -ForegroundColor Green
Write-Host ""

# Build image
Write-Host "[4/6] Construyendo imagen Docker..." -ForegroundColor Yellow
Write-Host "      Esto puede tomar varios minutos..." -ForegroundColor Gray
docker build -t cuda-lab-back:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Fallo el build de la imagen" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Imagen construida exitosamente" -ForegroundColor Green
Write-Host ""

# Run container
Write-Host "[5/6] Iniciando container..." -ForegroundColor Yellow
docker run -d `
    --name cuda-lab-backend `
    --gpus all `
    -p 5000:5000 `
    -e CUDA_VISIBLE_DEVICES=0 `
    --restart unless-stopped `
    cuda-lab-back:latest

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Fallo al iniciar el container" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Container iniciado" -ForegroundColor Green
Write-Host ""

# Wait for service to be ready
Write-Host "[6/6] Esperando que el servicio este listo..." -ForegroundColor Yellow
$maxRetries = 30
$retryCount = 0
$serviceReady = $false

while ($retryCount -lt $maxRetries) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.status -eq "ok") {
            $serviceReady = $true
            break
        }
    } catch {
        # Service not ready yet
    }
    
    Start-Sleep -Seconds 1
    $retryCount++
    Write-Host "." -NoNewline -ForegroundColor Gray
}

Write-Host ""

if ($serviceReady) {
    Write-Host "[OK] Servicio listo y funcionando" -ForegroundColor Green
} else {
    Write-Host "[WARNING] El servicio tardo mas de lo esperado. Verifica los logs." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "DEPLOYMENT COMPLETO" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Servicio disponible en: http://localhost:5000" -ForegroundColor Green
Write-Host ""
Write-Host "Comandos utiles:" -ForegroundColor Yellow
Write-Host "  Ver logs:        docker logs -f cuda-lab-backend" -ForegroundColor Gray
Write-Host "  Detener:         docker stop cuda-lab-backend" -ForegroundColor Gray
Write-Host "  Reiniciar:       docker restart cuda-lab-backend" -ForegroundColor Gray
Write-Host "  Eliminar:        docker rm -f cuda-lab-backend" -ForegroundColor Gray
Write-Host "  Ver GPU:         docker exec -it cuda-lab-backend nvidia-smi" -ForegroundColor Gray
Write-Host "  Entrar al cont:  docker exec -it cuda-lab-backend bash" -ForegroundColor Gray
Write-Host ""
Write-Host "Endpoints disponibles:" -ForegroundColor Yellow
Write-Host "  GET  http://localhost:5000/health" -ForegroundColor Gray
Write-Host "  GET  http://localhost:5000/filters" -ForegroundColor Gray
Write-Host "  POST http://localhost:5000/filters/gaussian" -ForegroundColor Gray
Write-Host "  POST http://localhost:5000/filters/prewitt" -ForegroundColor Gray
Write-Host "  POST http://localhost:5000/filters/cr7" -ForegroundColor Gray
Write-Host ""
