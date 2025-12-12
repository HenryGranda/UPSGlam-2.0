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

# ============================================================
# TECHNICAL DOCUMENTATION - REQUEST FLOW
# ============================================================
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "REQUEST FLOW & FILTER SPECIFICATIONS" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. REQUEST PATH THROUGH API GATEWAY:" -ForegroundColor Yellow
Write-Host "   Client -> Gateway (8080) -> CUDA Backend (5000)" -ForegroundColor Gray
Write-Host ""
Write-Host "   Gateway receives:" -ForegroundColor Cyan
Write-Host "     POST http://localhost:8080/api/filters/{filterName}" -ForegroundColor White
Write-Host "     Content-Type: image/jpeg" -ForegroundColor White
Write-Host "     Body: Binary image data" -ForegroundColor White
Write-Host ""
Write-Host "   Gateway rewrites to:" -ForegroundColor Cyan
Write-Host "     POST http://host.docker.internal:5000/filters/{filterName}" -ForegroundColor White
Write-Host "     (Gateway transforms /api/filters/* -> /filters/*)" -ForegroundColor Gray
Write-Host ""
Write-Host "   CUDA Backend receives:" -ForegroundColor Cyan
Write-Host "     POST http://localhost:5000/filters/{filterName}" -ForegroundColor White
Write-Host "     Content-Type: image/jpeg" -ForegroundColor White
Write-Host "     Body: Binary image data" -ForegroundColor White
Write-Host ""

Write-Host "2. FILTER SPECIFICATIONS & ARGUMENTS:" -ForegroundColor Yellow
Write-Host ""

Write-Host "   [gaussian] - Gaussian Blur Filter" -ForegroundColor Green
Write-Host "     Description:  Strong blur using Gaussian distribution" -ForegroundColor Gray
Write-Host "     CUDA Kernel:  gaussian_separable_u8 (two-pass: horizontal + vertical)" -ForegroundColor Gray
Write-Host "     Preset Args:  mask_size=121, sigma=auto (N/6)" -ForegroundColor Gray
Write-Host "     Block Dim:    (16, 16) threads" -ForegroundColor Gray
Write-Host "     Algorithm:    1D kernel separable convolution for efficiency" -ForegroundColor Gray
Write-Host ""

Write-Host "   [box_blur] - Box Blur Filter" -ForegroundColor Green
Write-Host "     Description:  Fast smoothing with simple averaging" -ForegroundColor Gray
Write-Host "     CUDA Kernel:  box_blur_u8" -ForegroundColor Gray
Write-Host "     Preset Args:  mask_size=91" -ForegroundColor Gray
Write-Host "     Block Dim:    (16, 16) threads" -ForegroundColor Gray
Write-Host "     Algorithm:    NxN box average convolution" -ForegroundColor Gray
Write-Host ""

Write-Host "   [prewitt] - Prewitt Edge Detection" -ForegroundColor Green
Write-Host "     Description:  Directional edge detection (first derivative)" -ForegroundColor Gray
Write-Host "     CUDA Kernel:  prewitt_u8 (Gx + Gy magnitude)" -ForegroundColor Gray
Write-Host "     Preset Args:  mask_size=3, gain=8.0" -ForegroundColor Gray
Write-Host "     Block Dim:    (16, 16) threads" -ForegroundColor Gray
Write-Host "     Algorithm:    Sobel-like gradient with 3x3 masks" -ForegroundColor Gray
Write-Host ""

Write-Host "   [laplacian] - Laplacian Edge Detection" -ForegroundColor Green
Write-Host "     Description:  Omnidirectional edge detection (second derivative)" -ForegroundColor Gray
Write-Host "     CUDA Kernel:  laplacian3x3_u8_to_u8 or conv_log_u8_to_f" -ForegroundColor Gray
Write-Host "     Preset Args:  mask_size=3 (classic) or N>3 (LoG)" -ForegroundColor Gray
Write-Host "     Block Dim:    (16, 16) threads" -ForegroundColor Gray
Write-Host "     Algorithm:    8-neighbor Laplacian or Laplacian of Gaussian" -ForegroundColor Gray
Write-Host ""

Write-Host "   [ups_logo] - UPS Logo Overlay with Aura" -ForegroundColor Green
Write-Host "     Description:  Creative filter with UPS branding and animated effects" -ForegroundColor Gray
Write-Host "     CUDA Kernel:  ups_logo_overlay_aura" -ForegroundColor Gray
Write-Host "     Preset Args:  mask_size=5 (gaussian blur background)" -ForegroundColor Gray
Write-Host "     Block Dim:    (16, 16) threads" -ForegroundColor Gray
Write-Host "     Algorithm:    Gaussian blur + PNG logo alpha blending + time-based aura" -ForegroundColor Gray
Write-Host "     Assets:       filters/assets/ups_logo_rgba.png" -ForegroundColor Gray
Write-Host ""

Write-Host "   [ups_color] - UPS Color Tint" -ForegroundColor Green
Write-Host "     Description:  Sepia tone with UPS corporate gold (#C69214)" -ForegroundColor Gray
Write-Host "     Implementation: CPU-based (no CUDA kernel)" -ForegroundColor Gray
Write-Host "     Preset Args:  mask_size=1 (no convolution)" -ForegroundColor Gray
Write-Host "     Algorithm:    Tone curve mapping with golden mid-tones" -ForegroundColor Gray
Write-Host "     Colors:       UPS Gold (#C69214), Maroon (#862633)" -ForegroundColor Gray
Write-Host ""

Write-Host "   [boomerang] - Boomerang Trail Effect" -ForegroundColor Green
Write-Host "     Description:  Trail effect with textured balls overlay" -ForegroundColor Gray
Write-Host "     CUDA Kernel:  draw_texture_balls" -ForegroundColor Gray
Write-Host "     Preset Args:  num_balls=8" -ForegroundColor Gray
Write-Host "     Block Dim:    (16, 16) threads" -ForegroundColor Gray
Write-Host "     Algorithm:    Trail generation + bilinear texture interpolation" -ForegroundColor Gray
Write-Host "     Assets:       filters/assets/sonrisa.png (crisp smile texture)" -ForegroundColor Gray
Write-Host ""

Write-Host "   [cr7] - CR7 Face Mask Overlay" -ForegroundColor Green
Write-Host "     Description:  Face detection and mask overlay" -ForegroundColor Gray
Write-Host "     CUDA Kernel:  alpha_blend_face" -ForegroundColor Gray
Write-Host "     Preset Args:  None (automatic face detection)" -ForegroundColor Gray
Write-Host "     Block Dim:    (16, 16) threads" -ForegroundColor Gray
Write-Host "     Algorithm:    OpenCV Haar Cascade + RGBA alpha blending" -ForegroundColor Gray
Write-Host "     Assets:       filters/assets/face_mask.png, haarcascade_frontalface_default.xml" -ForegroundColor Gray
Write-Host ""

Write-Host "3. CUDA COMPILATION:" -ForegroundColor Yellow
Write-Host "   Architecture: sm_89 (RTX 5070 Ti Blackwell)" -ForegroundColor Gray
Write-Host "   Method:       compile_cuda_kernel_to_ptx() -> nvcc -> PTX -> module_from_buffer()" -ForegroundColor Gray
Write-Host "   Reason:       Bypasses PyCUDA auto-detection to avoid sm_120 conflicts" -ForegroundColor Gray
Write-Host ""

Write-Host "4. RESPONSE FORMAT:" -ForegroundColor Yellow
Write-Host "   Content-Type: image/jpeg" -ForegroundColor Gray
Write-Host "   Headers:" -ForegroundColor Gray
Write-Host "     X-Filter-Applied: {filterName}" -ForegroundColor White
Write-Host "     Cache-Control: no-cache" -ForegroundColor White
Write-Host "   Body: Binary JPEG image data" -ForegroundColor Gray
Write-Host ""

Write-Host "5. EXAMPLE CURL COMMANDS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "   # Through Gateway (Production):" -ForegroundColor Cyan
Write-Host '   curl -X POST "http://localhost:8080/api/filters/gaussian" \' -ForegroundColor White
Write-Host '        -H "Content-Type: image/jpeg" \' -ForegroundColor White
Write-Host '        --data-binary "@" \' -ForegroundColor White
Write-Host '        -o "output_gaussian.jpinput.jpgg"' -ForegroundColor White
Write-Host ""
Write-Host "   # Direct to CUDA Backend (Development):" -ForegroundColor Cyan
Write-Host '   curl -X POST "http://localhost:5000/filters/cr7" \' -ForegroundColor White
Write-Host '        -H "Content-Type: image/jpeg" \' -ForegroundColor White
Write-Host '        --data-binary "@input.jpg" \' -ForegroundColor White
Write-Host '        -o "output_cr7.jpg"' -ForegroundColor White
Write-Host ""

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "END OF DOCUMENTATION" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
