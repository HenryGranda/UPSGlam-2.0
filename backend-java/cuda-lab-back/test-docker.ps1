# ============================================================
# Test CUDA Lab Backend Docker Container
# ============================================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "CUDA LAB BACKEND - TESTS" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$BASE_URL = "http://localhost:5000"

# Test 1: Health Check
Write-Host "TEST 1: Health Check" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/health" -Method GET
    if ($response.status -eq "ok") {
        Write-Host "[OK] Health check passed" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Unexpected health status: $($response.status)" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Health check failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 2: List Filters
Write-Host "TEST 2: List Available Filters" -ForegroundColor Yellow
try {
    $filters = Invoke-RestMethod -Uri "$BASE_URL/filters" -Method GET
    Write-Host "[OK] Found $($filters.filters.Count) filters:" -ForegroundColor Green
    foreach ($filter in $filters.filters) {
        Write-Host "  - $($filter.name): $($filter.description)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "[ERROR] Failed to list filters: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 3: Test Image Filter (if test image exists)
$testImagePath = ".\husky.jpg"
$outputDir = ".\tests"

# Create tests directory if it doesn't exist
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
    Write-Host "[INFO] Created output directory: $outputDir" -ForegroundColor Cyan
}

if (Test-Path $testImagePath) {
    Write-Host "TEST 3: Apply Gaussian Filter" -ForegroundColor Yellow
    try {
        $imageBytes = [System.IO.File]::ReadAllBytes($testImagePath)
        
        $response = Invoke-RestMethod `
            -Uri "$BASE_URL/filters/gaussian" `
            -Method POST `
            -ContentType "image/jpeg" `
            -Body $imageBytes `
            -OutFile "$outputDir\test_output_gaussian.jpg"
        
        if (Test-Path "$outputDir\test_output_gaussian.jpg") {
            Write-Host "[OK] Gaussian filter applied successfully" -ForegroundColor Green
            Write-Host "     Output saved to: $outputDir\test_output_gaussian.jpg" -ForegroundColor Cyan
        } else {
            Write-Host "[FAIL] Output file not created" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERROR] Filter test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host ""
    
    Write-Host "TEST 4: Apply Prewitt Filter" -ForegroundColor Yellow
    try {
        $imageBytes = [System.IO.File]::ReadAllBytes($testImagePath)
        
        $response = Invoke-RestMethod `
            -Uri "$BASE_URL/filters/prewitt" `
            -Method POST `
            -ContentType "image/jpeg" `
            -Body $imageBytes `
            -OutFile "$outputDir\test_output_prewitt.jpg"
        
        if (Test-Path "$outputDir\test_output_prewitt.jpg") {
            Write-Host "[OK] Prewitt filter applied successfully" -ForegroundColor Green
            Write-Host "     Output saved to: $outputDir\test_output_prewitt.jpg" -ForegroundColor Cyan
        } else {
            Write-Host "[FAIL] Output file not created" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERROR] Filter test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host ""
    
    Write-Host "TEST 5: Apply Laplacian Filter" -ForegroundColor Yellow
    try {
        $imageBytes = [System.IO.File]::ReadAllBytes($testImagePath)
        
        $response = Invoke-RestMethod `
            -Uri "$BASE_URL/filters/laplacian" `
            -Method POST `
            -ContentType "image/jpeg" `
            -Body $imageBytes `
            -OutFile "$outputDir\test_output_laplacian.jpg"
        
        if (Test-Path "$outputDir\test_output_laplacian.jpg") {
            Write-Host "[OK] Laplacian filter applied successfully" -ForegroundColor Green
            Write-Host "     Output saved to: $outputDir\test_output_laplacian.jpg" -ForegroundColor Cyan
        } else {
            Write-Host "[FAIL] Output file not created" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERROR] Filter test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host ""
    
    Write-Host "TEST 6: Apply Box Blur Filter" -ForegroundColor Yellow
    try {
        $imageBytes = [System.IO.File]::ReadAllBytes($testImagePath)
        
        $response = Invoke-RestMethod `
            -Uri "$BASE_URL/filters/box_blur" `
            -Method POST `
            -ContentType "image/jpeg" `
            -Body $imageBytes `
            -OutFile "$outputDir\test_output_box_blur.jpg"
        
        if (Test-Path "$outputDir\test_output_box_blur.jpg") {
            Write-Host "[OK] Box Blur filter applied successfully" -ForegroundColor Green
            Write-Host "     Output saved to: $outputDir\test_output_box_blur.jpg" -ForegroundColor Cyan
        } else {
            Write-Host "[FAIL] Output file not created" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERROR] Filter test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host ""
    
    Write-Host "TEST 7: Apply UPS Logo Filter" -ForegroundColor Yellow
    try {
        $imageBytes = [System.IO.File]::ReadAllBytes($testImagePath)
        
        $response = Invoke-RestMethod `
            -Uri "$BASE_URL/filters/ups_logo" `
            -Method POST `
            -ContentType "image/jpeg" `
            -Body $imageBytes `
            -OutFile "$outputDir\test_output_ups_logo.jpg"
        
        if (Test-Path "$outputDir\test_output_ups_logo.jpg") {
            Write-Host "[OK] UPS Logo filter applied successfully" -ForegroundColor Green
            Write-Host "     Output saved to: $outputDir\test_output_ups_logo.jpg" -ForegroundColor Cyan
        } else {
            Write-Host "[FAIL] Output file not created" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERROR] Filter test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host ""
    
    Write-Host "TEST 8: Apply UPS Color Filter" -ForegroundColor Yellow
    try {
        $imageBytes = [System.IO.File]::ReadAllBytes($testImagePath)
        
        $response = Invoke-RestMethod `
            -Uri "$BASE_URL/filters/ups_color" `
            -Method POST `
            -ContentType "image/jpeg" `
            -Body $imageBytes `
            -OutFile "$outputDir\test_output_ups_color.jpg"
        
        if (Test-Path "$outputDir\test_output_ups_color.jpg") {
            Write-Host "[OK] UPS Color filter applied successfully" -ForegroundColor Green
            Write-Host "     Output saved to: $outputDir\test_output_ups_color.jpg" -ForegroundColor Cyan
        } else {
            Write-Host "[FAIL] Output file not created" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERROR] Filter test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host ""
    
    Write-Host "TEST 9: Apply Boomerang Filter" -ForegroundColor Yellow
    try {
        $imageBytes = [System.IO.File]::ReadAllBytes($testImagePath)
        
        $response = Invoke-RestMethod `
            -Uri "$BASE_URL/filters/boomerang" `
            -Method POST `
            -ContentType "image/jpeg" `
            -Body $imageBytes `
            -OutFile "$outputDir\test_output_boomerang.jpg"
        
        if (Test-Path "$outputDir\test_output_boomerang.jpg") {
            Write-Host "[OK] Boomerang filter applied successfully" -ForegroundColor Green
            Write-Host "     Output saved to: $outputDir\test_output_boomerang.jpg" -ForegroundColor Cyan
        } else {
            Write-Host "[FAIL] Output file not created" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERROR] Filter test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host ""
    
    Write-Host "TEST 10: Apply CR7 Filter (Face Mask)" -ForegroundColor Yellow
    try {
        $imageBytes = [System.IO.File]::ReadAllBytes($testImagePath)
        
        $response = Invoke-RestMethod `
            -Uri "$BASE_URL/filters/cr7" `
            -Method POST `
            -ContentType "image/jpeg" `
            -Body $imageBytes `
            -OutFile "$outputDir\test_output_cr7.jpg"
        
        if (Test-Path "$outputDir\test_output_cr7.jpg") {
            Write-Host "[OK] CR7 filter applied successfully" -ForegroundColor Green
            Write-Host "     Output saved to: $outputDir\test_output_cr7.jpg" -ForegroundColor Cyan
        } else {
            Write-Host "[FAIL] Output file not created" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERROR] Filter test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "TEST 3-10: Skipped (no test image found at $testImagePath)" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "TESTS COMPLETED" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para ver logs del container:" -ForegroundColor Yellow
Write-Host "  docker logs -f cuda-lab-backend" -ForegroundColor Gray
Write-Host ""
