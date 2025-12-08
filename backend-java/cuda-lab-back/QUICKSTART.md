# CUDA Image Lab - Backend: Guía de inicio rápido

## 🚀 Iniciar el servidor

### Opción 1: Desarrollo local (recomendado)
```bash
# Con recarga automática
python -m uvicorn app:app --host 0.0.0.0 --port 5000 --reload

# Sin recarga automática
python -m uvicorn app:app --host 0.0.0.0 --port 5000
```

### Opción 2: Docker
```bash
docker-compose up
```

## 🧪 Probar los endpoints

### Endpoint de salud
```bash
curl http://localhost:5000/health
```

### Listar filtros disponibles
```bash
curl http://localhost:5000/filters
```

### Aplicar filtro a una imagen

#### Gaussian Blur
```bash
curl.exe -X POST "http://localhost:5000/filters/gaussian" -H "Content-Type: image/jpeg" --data-binary "@husky.jpg" -o "husky_gaussian.jpg"
```

#### Box Blur
```bash
curl.exe -X POST "http://localhost:5000/filters/box_blur" -H "Content-Type: image/jpeg" --data-binary "@husky.jpg" -o "husky_box_blur.jpg"
```

#### Prewitt (Edge Detection)
```bash
curl.exe -X POST "http://localhost:5000/filters/prewitt" -H "Content-Type: image/jpeg" --data-binary "@husky.jpg" -o "husky_prewitt.jpg"
```

#### Laplacian (Edge Detection)
```bash
curl.exe -X POST "http://localhost:5000/filters/laplacian" -H "Content-Type: image/jpeg" --data-binary "@husky.jpg" -o "husky_laplacian.jpg"
```

#### UPS Logo
```bash
curl.exe -X POST "http://localhost:5000/filters/ups_logo" -H "Content-Type: image/jpeg" --data-binary "@husky.jpg" -o "husky_ups_logo.jpg"
```

#### UPS Color
```bash
curl.exe -X POST "http://localhost:5000/filters/ups_color" -H "Content-Type: image/jpeg" --data-binary "@husky.jpg" -o "husky_ups_color.jpg"
```

## 🐛 Solución de problemas

### Imagen corrupta / error al abrir
**Causa:** Puerto incorrecto (5000 vs 8000)  
### Imagen corrupta / error al abrir
**Causa:** El servidor no está corriendo o puerto incorrecto  
**Solución:** Asegúrate de usar `localhost:5000` en todos los comandos

### Connection refused
**Causa:** El servidor no está corriendo  
**Solución:** Inicia el servidor con `python -m uvicorn app:app --host 0.0.0.0 --port 5000`
**Causa:** GPU no disponible o drivers desactualizados  
**Solución:** Verifica que tienes una GPU NVIDIA con drivers CUDA instalados

## 📝 Notas

- Las imágenes se procesan en RGB (color preservado)
- Formato de salida: JPEG con calidad 95
- Configuraciones de filtros son preestablecidas (no requieren parámetros)
