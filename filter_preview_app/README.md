# Filter Preview App

Mini app Flutter para preview de filtros en tiempo real y captura con procesamiento en backend.

## 🎯 Funcionalidad

1. **Preview en Tiempo Real**: Aplica filtros ligeros localmente (simulando los del backend)
2. **Captura**: Toma foto original (sin filtro)
3. **Procesamiento Backend**: Envía al backend cuda-lab-back para procesamiento PyCUDA de alta calidad

## 📱 Filtros Disponibles

- Gaussian Blur
- Box Blur
- Prewitt (Edge Detection)
- Laplacian (Edge Detection)
- UPS Logo (simulado en preview, real en backend)
- UPS Color
- Boomerang (simulado en preview, real en backend)

## 🚀 Instalación

```bash
cd filter_preview_app
flutter pub get
flutter run
```

## 🔧 Configuración Backend

Edita `lib/services/filter_service.dart` y cambia la URL del backend:

```dart
static const String baseUrl = 'http://TU_IP:5000';  // Cambia esto
```

## 📦 Dependencias

- `camera`: Acceso a la cámara del dispositivo
- `image`: Procesamiento de imágenes para preview de filtros
- `http`: Comunicación con backend
- `path_provider`: Almacenamiento temporal

## 🎨 Arquitectura

```
Flutter App
├── Camera Preview (con filtros locales ligeros)
├── Selector de filtros (botones)
├── Botón de captura
└── Envío al backend
    └── Backend cuda-lab-back (PyCUDA)
        └── Retorna imagen procesada de alta calidad
```

## 🔥 Uso

1. Abre la app
2. Selecciona un filtro (verás preview en tiempo real)
3. Toma la foto
4. La imagen ORIGINAL se envía al backend
5. Backend procesa con PyCUDA
6. Recibes imagen final de alta calidad
