# ============================================================
# Build and Run API Gateway Docker Container
# ============================================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "API GATEWAY - DOCKER BUILD & RUN" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$IMAGE_NAME = "upsglam-api-gateway"
$CONTAINER_NAME = "api-gateway"
$PORT = "8080"

# Step 1: Stop and remove existing container if running
Write-Host "Step 1: Cleaning up existing container..." -ForegroundColor Yellow
$existingContainer = docker ps -aq -f name=$CONTAINER_NAME
if ($existingContainer) {
    Write-Host "  - Stopping container: $CONTAINER_NAME" -ForegroundColor Gray
    docker stop $CONTAINER_NAME | Out-Null
    Write-Host "  - Removing container: $CONTAINER_NAME" -ForegroundColor Gray
    docker rm $CONTAINER_NAME | Out-Null
    Write-Host "  [OK] Container removed" -ForegroundColor Green
} else {
    Write-Host "  [INFO] No existing container found" -ForegroundColor Cyan
}
Write-Host ""

# Step 2: Build Docker image
Write-Host "Step 2: Building Docker image..." -ForegroundColor Yellow
Write-Host "  Image: $IMAGE_NAME" -ForegroundColor Cyan
Write-Host ""

docker build -t ${IMAGE_NAME}:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  [OK] Image built successfully" -ForegroundColor Green
Write-Host ""

# Step 3: Run the container
Write-Host "Step 3: Starting container..." -ForegroundColor Yellow
Write-Host "  Container: $CONTAINER_NAME" -ForegroundColor Cyan
Write-Host "  Port: $PORT" -ForegroundColor Cyan
Write-Host ""

$containerId = docker run -d `
    --name $CONTAINER_NAME `
    -p ${PORT}:8080 `
    --restart unless-stopped `
    ${IMAGE_NAME}:latest

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Failed to start container!" -ForegroundColor Red
    exit 1
}

Write-Host "  [OK] Container started: $containerId" -ForegroundColor Green
Write-Host ""

# Step 4: Wait for service to be ready
Write-Host "Step 4: Waiting for service to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check container status
$status = docker ps --filter "name=$CONTAINER_NAME" --format "{{.Status}}"
if ($status) {
    Write-Host "  [OK] Container is running: $status" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Container may not be running" -ForegroundColor Yellow
    Write-Host "  Check logs with: docker logs $CONTAINER_NAME" -ForegroundColor Gray
}
Write-Host ""

# Step 5: Test health endpoint
Write-Host "Step 5: Testing health endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:${PORT}/actuator/health" -Method GET -TimeoutSec 10
    if ($response.status -eq "UP") {
        Write-Host "  [OK] API Gateway is UP and healthy!" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] Health check returned: $($response.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [WARNING] Health check failed (service may still be starting): $($_.Exception.Message)" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "API GATEWAY DEPLOYED SUCCESSFULLY" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Container Info:" -ForegroundColor Yellow
Write-Host "  Name:     $CONTAINER_NAME" -ForegroundColor Cyan
Write-Host "  Image:    ${IMAGE_NAME}:latest" -ForegroundColor Cyan
Write-Host "  Port:     http://localhost:${PORT}" -ForegroundColor Cyan
Write-Host ""
Write-Host "Useful Commands:" -ForegroundColor Yellow
Write-Host "  View logs:      docker logs -f $CONTAINER_NAME" -ForegroundColor Gray
Write-Host "  Stop:           docker stop $CONTAINER_NAME" -ForegroundColor Gray
Write-Host "  Restart:        docker restart $CONTAINER_NAME" -ForegroundColor Gray
Write-Host "  Remove:         docker rm -f $CONTAINER_NAME" -ForegroundColor Gray
Write-Host ""
Write-Host "API Endpoints:" -ForegroundColor Yellow
Write-Host "  Health:         http://localhost:${PORT}/actuator/health" -ForegroundColor Gray
Write-Host "  Gateway Routes: http://localhost:${PORT}/actuator/gateway/routes" -ForegroundColor Gray
Write-Host "  Auth Service:   http://localhost:${PORT}/api/auth/*" -ForegroundColor Gray
Write-Host "  Post Service:   http://localhost:${PORT}/api/posts/*" -ForegroundColor Gray
Write-Host "  CUDA Service:   http://localhost:${PORT}/api/filters/*" -ForegroundColor Gray
Write-Host ""
