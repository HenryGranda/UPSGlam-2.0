# Regla de firewall mas permisiva para permitir TODAS las conexiones entrantes en los puertos
# Ejecutar como Administrador

Write-Host "Configurando regla permisiva de firewall..." -ForegroundColor Cyan
Write-Host ""

# Eliminar reglas anteriores si existen
$rulesToRemove = @(
    "UPSGlam API Gateway",
    "UPSGlam Post Service",
    "UPSGlam Auth Service",
    "UPSGlam CUDA Backend"
)

foreach ($rule in $rulesToRemove) {
    try {
        Remove-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue
        Write-Host ("Regla '{0}' eliminada" -f $rule) -ForegroundColor Yellow
    } catch {
        # Regla no existe, continuar
    }
}

Write-Host ""
Write-Host "Creando nueva regla PERMISIVA..." -ForegroundColor Green

# Crear UNA regla para todos los puertos
New-NetFirewallRule `
    -DisplayName "UPSGlam Backend (Todos los puertos)" `
    -Direction Inbound `
    -LocalPort 8080,8081,8082,5000 `
    -Protocol TCP `
    -Action Allow `
    -Profile Any `
    -Enabled True `
    -RemoteAddress Any

Write-Host ""
Write-Host "Regla creada: Puertos 8080, 8081, 8082 y 5000 ABIERTOS para TODAS las conexiones" -ForegroundColor Green
Write-Host ""

Write-Host "Ahora prueba desde el navegador de tu telefono:" -ForegroundColor Cyan
Write-Host "  http://192.168.1.252:8080/health" -ForegroundColor Yellow
Write-Host ""

Write-Host 'Deberia responder: {"status":"UP"}' -ForegroundColor White
