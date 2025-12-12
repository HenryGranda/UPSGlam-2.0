# 📱 UPSGlam Mobile App - Flutter

## 📋 Descripción

Aplicación móvil de UPSGlam 2.0 desarrollada con **Flutter 3.10**. Red social de fotos estilo Instagram con filtros CUDA, autenticación Firebase y funcionalidades en tiempo real.

---

## 🏗️ Stack Tecnológico

- **Framework**: Flutter 3.10.1
- **Lenguaje**: Dart 3.10.1
- **State Management**: setState (built-in)
- **HTTP Client**: http package
- **Image Handling**: image_picker, camera
- **Storage**: shared_preferences
- **UI**: Material Design 3

---

## ✨ Características Principales

### 1. **Autenticación**
- Login con email/password
- Registro de nuevos usuarios
- Persistencia de sesión (SharedPreferences)
- Logout

### 2. **Feed de Posts**
- Vista de publicaciones en tiempo real
- Scroll infinito
- Pull-to-refresh
- Like/Unlike posts
- Contador de likes y comentarios

### 3. **Crear Publicaciones**
- Captura de foto con cámara
- Selección de galería
- Preview en vivo con filtros locales (Dart)
- Aplicación de filtros CUDA (GPU Backend)
- Descripción de post
- Upload a Supabase Storage

### 4. **Filtros**

#### Filtros Locales (Preview en Vivo)
- Procesamiento en Dart
- Vista previa en tiempo real con cámara
- No requiere backend

#### Filtros CUDA (GPU)
- Gaussian Blur
- Box Blur
- Prewitt Edge Detection
- Laplacian Edge Detection
- UPS Logo Overlay
- Boomerang Effect
- CR7 Mask

### 5. **Perfil de Usuario**
- Ver perfil propio
- Ver perfil de otros usuarios
- Grid de posts del usuario
- Contador de posts/seguidores/seguidos
- Follow/Unfollow

### 6. **Comentarios**
- Agregar comentarios a posts
- Ver lista de comentarios
- Timestamp de comentarios

---

## 📁 Estructura del Proyecto

```
mobile_app/
├── lib/
│   ├── main.dart                    # Entry point
│   ├── models/
│   │   ├── current_user.dart       # Modelo de usuario actual
│   │   ├── post_model.dart         # Modelo de post
│   │   └── comment_model.dart      # Modelo de comentario
│   ├── screens/
│   │   ├── auth/
│   │   │   ├── login_screen.dart
│   │   │   └── register_screen.dart
│   │   ├── home/
│   │   │   ├── home_screen.dart
│   │   │   ├── feed_view.dart
│   │   │   ├── create_post_view.dart
│   │   │   └── live_preview_panel.dart
│   │   └── profile/
│   │       └── profile_screen.dart
│   ├── services/
│   │   ├── auth_service.dart       # API de autenticación
│   │   ├── post_service.dart       # API de posts
│   │   ├── filter_service.dart     # API de filtros CUDA
│   │   └── storage_service.dart    # SharedPreferences
│   └── widgets/
│       ├── post_card.dart          # Card de post en feed
│       └── common/                 # Widgets reutilizables
├── assets/
│   ├── images/
│   │   └── logoups.png
│   └── avatars/
├── android/                         # Configuración Android
├── ios/                            # Configuración iOS
├── test/                           # Tests
├── pubspec.yaml                    # Dependencias
└── README.md
```

---

## 🔧 Configuración

### Dependencias Principales (`pubspec.yaml`)

```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # Image handling
  image_picker: ^1.1.0
  camera: ^0.10.5+5
  image: ^4.1.3
  
  # Permissions
  permission_handler: ^11.0.1
  
  # Storage
  shared_preferences: ^2.5.3
  
  # Utils
  intl: ^0.19.0
  http: ^1.2.0
  path_provider: ^2.1.1
  
  # Filter preview
  filter_preview_app:
    path: ../filter_preview_app
```

### API Configuration

Editar las URLs del backend en los servicios:

```dart
// lib/services/auth_service.dart
static const String baseUrl = 'http://10.0.2.2:8080/api';  // Android Emulator
// static const String baseUrl = 'http://localhost:8080/api';  // iOS Simulator
// static const String baseUrl = 'http://192.168.1.100:8080/api';  // Dispositivo físico

// lib/services/filter_service.dart
static const String cudaBaseUrl = 'http://10.0.2.2:5000';
```

---

## 🚀 Instalación y Ejecución

### Prerrequisitos

```bash
# Flutter SDK
flutter --version  # >= 3.10.1

# Verificar instalación
flutter doctor

# Dependencias del proyecto
flutter pub get
```

### Ejecución

```bash
# Ejecutar en emulador/dispositivo
flutter run

# Modo release
flutter run --release

# Especificar dispositivo
flutter devices
flutter run -d <device-id>

# Hot reload está activo por defecto
# Presiona 'r' para hot reload
# Presiona 'R' para hot restart
```

### Build

```bash
# Android APK
flutter build apk --release

# Android App Bundle (para Play Store)
flutter build appbundle --release

# iOS (requiere Mac)
flutter build ios --release
```

---

## 🔗 Integración con Backend

### API Gateway (Port 8080)

La app se comunica con el API Gateway que enruta a los microservicios:

```
Mobile App
    ↓
API Gateway (8080)
    ├── /api/auth/** → Auth Service (8082)
    ├── /api/posts/** → Post Service (8081)
    ├── /api/feed/** → Post Service (8081)
    ├── /api/images/** → Post Service (8081)
    └── /api/filters/** → CUDA Backend (5000)
```

### Auth Headers

```dart
// Todas las requests autenticadas incluyen:
headers: {
  'X-User-Id': currentUserId,
  'X-Username': currentUsername,
  'Content-Type': 'application/json',
}
```

---

## 📱 Pantallas Principales

### 1. Login Screen
- Email/password input
- Botón de login
- Link a registro
- Validación de campos

### 2. Register Screen
- Email, username, password inputs
- Validación de formato
- Creación de cuenta
- Navegación a login después de registro exitoso

### 3. Home Screen (Bottom Navigation)
- **Feed**: Lista de posts
- **Create**: Crear nueva publicación
- **Profile**: Perfil del usuario

### 4. Feed View
- Lista scrolleable de posts
- Pull-to-refresh
- Like/Unlike
- Ver comentarios
- Avatar y username clicables

### 5. Create Post View
- Toggle: Preview en vivo vs. Foto capturada
- Captura con cámara o galería
- Selección de filtros (local o CUDA)
- Campo de descripción
- Botón de publicar

### 6. Profile Screen
- Header con avatar, nombre, bio
- Estadísticas (posts/followers/following)
- Botón Follow/Unfollow (si no es tu perfil)
- Grid de posts del usuario

---

## 🎨 Filtros

### Filtros Locales (Dart - Preview en Vivo)
Implementados en `filter_preview_app`:
- Aplicación en tiempo real con cámara
- Procesamiento en CPU (Dart)
- No requieren backend

### Filtros CUDA (GPU - Procesamiento Final)
```dart
// Aplicar filtro CUDA
final filteredPath = await FilterService.instance.applyFilter(
  imageFile: File(imagePath),
  filterName: 'gaussian',
);
```

Filtros disponibles:
- `gaussian`: Gaussian Blur
- `box_blur`: Box Blur
- `prewitt`: Prewitt Edge Detection
- `laplacian`: Laplacian Edge Detection
- `ups_logo`: UPS Logo Overlay
- `boomerang`: Boomerang Effect
- `cr7`: CR7 Mask

---

## 🧪 Testing

```bash
# Ejecutar todos los tests
flutter test

# Test con coverage
flutter test --coverage

# Test específico
flutter test test/services/auth_service_test.dart
```

---

## 📊 Performance

### Optimizaciones Implementadas
- ✅ Caché de imágenes (CachedNetworkImage)
- ✅ Lazy loading en feed
- ✅ Compresión de imágenes antes de upload
- ✅ Debounce en búsquedas
- ✅ setState mínimo y eficiente

### Memory Management
- Dispose de controllers
- Limpieza de listeners
- Gestión de streams

---

## 🐛 Troubleshooting

### Error: Cannot connect to backend

```dart
// Verificar URL según plataforma:

// Android Emulator
baseUrl = 'http://10.0.2.2:8080/api';

// iOS Simulator  
baseUrl = 'http://localhost:8080/api';

// Dispositivo físico (misma red)
baseUrl = 'http://192.168.1.100:8080/api';
```

### Error: Camera permission denied

```yaml
# android/app/src/main/AndroidManifest.xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />

# ios/Runner/Info.plist
<key>NSCameraUsageDescription</key>
<string>Necesitamos acceso a la cámara para tomar fotos</string>
```

### Error: Build fails

```bash
# Limpiar y rebuild
flutter clean
flutter pub get
flutter run
```

---

## 📚 Referencias

- [Flutter Documentation](https://docs.flutter.dev/)
- [Dart Language Tour](https://dart.dev/guides/language/language-tour)
- [Material Design 3](https://m3.material.io/)
- [Backend API Documentation](../backend-java/api-gateway/API-ROUTES.md)

---

## 📝 Changelog

### Version 1.0.0
- ✅ Autenticación con Firebase
- ✅ Feed de posts en tiempo real
- ✅ Creación de posts con filtros CUDA
- ✅ Sistema de likes
- ✅ Sistema de comentarios
- ✅ Perfiles de usuario
- ✅ Follow/Unfollow
- ✅ Preview de filtros en vivo

---

## 👥 Autor

**UPSGlam Development Team**  
Universidad Politécnica Salesiana  
Quito, Ecuador

---

## 📄 Licencia

Este proyecto es privado y confidencial.  
© 2025 Universidad Politécnica Salesiana. Todos los derechos reservados.
