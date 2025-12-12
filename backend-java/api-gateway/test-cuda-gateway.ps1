# ============================================================
# Test CUDA Service through API Gateway (Port 8080)
# ============================================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "CUDA SERVICE - GATEWAY TESTS (Port 8080)" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$GATEWAY_URL = "http://localhost:8080"
$testImagePath = "C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\cuda-lab-back\husky.jpg"
$outputDir = "C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\cuda-lab-back\tests\gateway_outputs"

# Create output directory if it doesn't exist
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
    Write-Host "[INFO] Created output directory: $outputDir" -ForegroundColor Cyan
}

# Test 1: Health Check through Gateway
Write-Host "TEST 1: CUDA Service Health Check (through Gateway)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$GATEWAY_URL/api/health/cuda" -Method GET
    if ($response.status -eq "ok") {
        Write-Host "[OK] Health check passed through Gateway" -ForegroundColor Green
        Write-Host "     Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Cyan
    } else {
        Write-Host "[FAIL] Unexpected health status: $($response.status)" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Health check failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 2: List Filters through Gateway
Write-Host "TEST 2: List Available Filters (through Gateway)" -ForegroundColor Yellow
try {
    $filters = Invoke-RestMethod -Uri "$GATEWAY_URL/api/filters" -Method GET
    Write-Host "[OK] Found $($filters.filters.Count) filters through Gateway:" -ForegroundColor Green
    foreach ($filter in $filters.filters) {
        Write-Host "  - $($filter.name): $($filter.description)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "[ERROR] Failed to list filters: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 3-10: Apply filters through Gateway
if (Test-Path $testImagePath) {
    $filterTests = @(
        @{ Name = "gaussian"; Display = "Gaussian Blur" },
        @{ Name = "prewitt"; Display = "Prewitt Edge Detection" },
        @{ Name = "laplacian"; Display = "Laplacian Edge Detection" },
        @{ Name = "box_blur"; Display = "Box Blur" },
        @{ Name = "ups_logo"; Display = "UPS Logo Overlay" },
        @{ Name = "ups_color"; Display = "UPS Color Tint" },
        @{ Name = "boomerang"; Display = "Boomerang Effect" },
        @{ Name = "cr7"; Display = "CR7 Face Mask" }
    )
    
    $testNumber = 3
    foreach ($filter in $filterTests) {
        Write-Host "TEST $testNumber`: Apply $($filter.Display) Filter (through Gateway)" -ForegroundColor Yellow
        try {
            $imageBytes = [System.IO.File]::ReadAllBytes($testImagePath)
            
            $response = Invoke-RestMethod `
                -Uri "$GATEWAY_URL/api/filters/$($filter.Name)" `
                -Method POST `
                -ContentType "image/jpeg" `
                -Body $imageBytes `
                -OutFile "$outputDir\gateway_output_$($filter.Name).jpg"
            
            if (Test-Path "$outputDir\gateway_output_$($filter.Name).jpg") {
                $fileInfo = Get-Item "$outputDir\gateway_output_$($filter.Name).jpg"
                $fileSizeKB = [math]::Round($fileInfo.Length / 1KB, 2)
                Write-Host "[OK] $($filter.Display) filter applied successfully through Gateway" -ForegroundColor Green
                Write-Host "     Output: $outputDir\gateway_output_$($filter.Name).jpg ($fileSizeKB KB)" -ForegroundColor Cyan
            } else {
                Write-Host "[FAIL] Output file not created" -ForegroundColor Red
            }
        } catch {
            Write-Host "[ERROR] Filter test failed: $($_.Exception.Message)" -ForegroundColor Red
        }
        Write-Host ""
        $testNumber++
    }
} else {
    Write-Host "TEST 3-10: Skipped (no test image found at $testImagePath)" -ForegroundColor Yellow
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "GATEWAY TESTS COMPLETED" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Comparison:" -ForegroundColor Yellow
Write-Host "  Direct CUDA:  http://localhost:5000/filters/{filterName}" -ForegroundColor Gray
Write-Host "  Via Gateway:  http://localhost:8080/api/filters/{filterName}" -ForegroundColor Gray
Write-Host ""
Write-Host "All images saved to: $outputDir" -ForegroundColor Cyan
Write-Host ""
