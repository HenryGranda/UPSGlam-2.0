# Test de conectividad desde otro dispositivo
# Ejecuta esto desde PowerShell en tu PC

Write-Host "Verificando que los servicios esten accesibles desde la red local..." -ForegroundColor Cyan
Write-Host ""

# Obtener IP local (privada)
$ipAddress = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.PrefixOrigin -in @("Dhcp","Manual") } |
    Where-Object { $_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" -or $_.IPAddress -like "172.16.*" -or $_.IPAddress -like "172.17.*" -or $_.IPAddress -like "172.18.*" -or $_.IPAddress -like "172.19.*" -or $_.IPAddress -like "172.2*.*" -or $_.IPAddress -like "172.3*.*" }
).IPAddress | Select-Object -First 1

if (-not $ipAddress) {
    Write-Host "No se pudo detectar tu IP local" -ForegroundColor Red
    Write-Host "Ejecuta: ipconfig" -ForegroundColor Yellow
    exit 1
}

Write-Host ("Tu IP local es: {0}" -f $ipAddress) -ForegroundColor Green
Write-Host ""

# Verificar que Docker corre
Write-Host "Verificando contenedores Docker..." -ForegroundColor Yellow
$containers = docker ps --format "{{.Names}} - {{.Status}}"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker no esta corriendo o no hay contenedores activos" -ForegroundColor Red
    exit 1
}

Write-Host $containers
Write-Host ""

# Test de puertos locales
Write-Host "Testeando puertos localmente..." -ForegroundColor Yellow

$ports = @(8080, 8081, 8082, 5000)
$allOk = $true

foreach ($port in $ports) {
    try {
        $null = Invoke-WebRequest -Uri ("http://localhost:{0}/health" -f $port) -TimeoutSec 5 -ErrorAction Stop
        Write-Host ("Puerto {0} - Responde OK" -f $port) -ForegroundColor Green
    } catch {
        Write-Host ("Puerto {0} - No responde" -f $port) -ForegroundColor Red
        $allOk = $false
    }
}

Write-Host ""

if ($allOk) {
    Write-Host "Todos los servicios funcionan localmente" -ForegroundColor Green
    Write-Host ""
    Write-Host "Para conectar desde tu telefono, usa esta IP en la app:" -ForegroundColor Cyan
    Write-Host ("  {0}" -f $ipAddress) -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Asegurate que:" -ForegroundColor Yellow
    Write-Host "  1. Tu telefono esta en la MISMA red WiFi" -ForegroundColor White
    Write-Host "  2. El firewall de Windows permite conexiones entrantes" -ForegroundColor White
    Write-Host "  3. No hay VPN activa en tu PC o telefono" -ForegroundColor White
    Write-Host ""
    Write-Host "Test desde el telefono (navegador):" -ForegroundColor Cyan
    Write-Host ("  http://{0}:8080/health" -f $ipAddress) -ForegroundColor Yellow
    Write-Host 'Deberia mostrar algo como: {"status":"UP"}' -ForegroundColor White
}
else {
    Write-Host "Algunos servicios no estan funcionando" -ForegroundColor Red
    Write-Host "Ejecuta: docker-compose up -d" -ForegroundColor Yellow
}
