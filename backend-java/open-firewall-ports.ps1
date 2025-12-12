# Script para abrir puertos del backend en Windows Firewall
# Ejecutar como Administrador

Write-Host "Abriendo puertos en Windows Firewall..." -ForegroundColor Cyan

# API Gateway (8080)
New-NetFirewallRule -DisplayName "UPSGlam API Gateway" `
    -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow -Profile Any

# Post Service (8081)
New-NetFirewallRule -DisplayName "UPSGlam Post Service" `
    -Direction Inbound -LocalPort 8081 -Protocol TCP -Action Allow -Profile Any

# Auth Service (8082)
New-NetFirewallRule -DisplayName "UPSGlam Auth Service" `
    -Direction Inbound -LocalPort 8082 -Protocol TCP -Action Allow -Profile Any

# CUDA Backend (5000)
New-NetFirewallRule -DisplayName "UPSGlam CUDA Backend" `
    -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow -Profile Any

Write-Host "Puertos abiertos correctamente." -ForegroundColor Green
