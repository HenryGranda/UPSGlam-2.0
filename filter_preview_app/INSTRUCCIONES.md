# 📱 Instrucciones de Uso - Filter Preview App

## 🚀 Instalación y Configuración

### 1. Instalar Dependencias

```bash
cd filter_preview_app
flutter pub get
```

### 2. Configurar URL del Backend

Edita `lib/services/filter_service.dart`:

```dart
static const String baseUrl = 'http://TU_IP:5000';
```

**Opciones según tu caso:**

- **Emulador Android**: `http://10.0.2.2:5000`
- **Dispositivo Real (misma red WiFi)**: `http://192.168.X.X:5000`
  - Encuentra tu IP con: `ipconfig` (Windows) o `ifconfig` (Linux/Mac)
- **Servidor remoto**: `http://TU_SERVIDOR:5000`

### 3. Asegurar que el Backend esté Corriendo

```bash
cd C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\cuda-lab-back
python -m uvicorn app:app --host 0.0.0.0 --port 5000
```

### 4. Ejecutar la App

```bash
# Para Android
flutter run

# Para dispositivo específico
flutter devices
flutter run -d DEVICE_ID
```

## 🎯 Flujo de Uso

1. **Abrir App** → Se activa la cámara
2. **Seleccionar Filtro** → Desplazar horizontalmente los filtros en la parte inferior
3. **Tomar Foto** → Presionar el botón circular dorado
4. **Procesamiento** → La imagen ORIGINAL se envía al backend
5. **Ver Resultado** → Imagen procesada con PyCUDA de alta calidad

## 🎨 Filtros Disponibles

| Filtro | Descripción | Tipo |
|--------|-------------|------|
| Gaussian | Suavizado gaussiano fuerte | Convolución |
| Box Blur | Suavizado rápido | Convolución |
| Prewitt | Detección de bordes direccional | Convolución |
| Laplacian | Detección de bordes omnidireccional | Convolución |
| UPS Logo | Logo Don Bosco con efectos de aura | Creativo |
| UPS Color | Tinte con colores corporativos UPS | Creativo |
| Boomerang | Rastro de bolas texturizadas | Creativo |

## 🔧 Solución de Problemas

### Error: "No se encontró ninguna cámara"
- Verifica permisos en: Configuración → Apps → UPS Glam Filters → Permisos → Cámara

### Error: "Timeout: El servidor tardó demasiado"
- Verifica que el backend esté corriendo
- Verifica que la URL sea correcta
- Verifica que el dispositivo y servidor estén en la misma red

### Error: "Error al procesar la imagen"
- Revisa logs del backend: `python -m uvicorn app:app --host 0.0.0.0 --port 5000`
- Verifica que PyCUDA esté instalado y funcionando

### Preview se ve diferente al resultado final
- ✅ **Esto es NORMAL y ESPERADO**
- Preview: Filtro simulado ligero (solo para visualización)
- Resultado: Filtro PyCUDA de alta calidad (procesamiento real)

## 📝 Notas Importantes

1. **Imagen Original**: La app SIEMPRE envía la imagen original sin filtro al backend
2. **Preview Local**: El preview en tiempo real es solo visual (no se aplica realmente)
3. **Procesamiento Real**: El procesamiento real se hace en el backend con PyCUDA
4. **Calidad**: La imagen final tiene MUCHO mejor calidad que el preview

## 🎓 Para Desarrollo

### Agregar Nuevo Filtro

1. Agrega el filtro en el backend (`cuda-lab-back/filters/`)
2. Registra en `app.py`
3. Agrega a la lista en `filter_selector.dart`:

```dart
{'id': 'mi_filtro', 'name': 'Mi Filtro', 'icon': Icons.star},
```

### Modificar Preview Local (Opcional)

Edita `camera_screen.dart` para aplicar filtros locales simulados antes de capturar.

## 📱 Capturas de Pantalla

```
[Cámara Activa]
   ↓
[Seleccionar Filtro]
   ↓
[Ver Preview Simulado]
   ↓
[Tomar Foto] ← Envía imagen ORIGINAL
   ↓
[Backend PyCUDA Procesa]
   ↓
[Ver Resultado Final]
```

## 🔥 Próximos Pasos (Opcional)

- [ ] Implementar preview local con filtros reales (usando `image` package)
- [ ] Agregar función de guardar en galería
- [ ] Compartir en redes sociales
- [ ] Historial de fotos procesadas
- [ ] Modo batch (procesar múltiples fotos)
