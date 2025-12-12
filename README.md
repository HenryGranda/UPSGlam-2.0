# 🎨 UPSGlam 2.0 - Social Media con Filtros GPU

## 📋 Descripción

**UPSGlam 2.0** es una red social de fotografía estilo Instagram desarrollada con **arquitectura de microservicios**, que permite a los usuarios capturar fotos, aplicar **filtros procesados en GPU con CUDA**, y compartirlas en un feed social con funcionalidades de likes, comentarios y sistema de seguimiento.

### 🎯 Objetivos del Proyecto

- ✅ Implementar arquitectura de microservicios con Spring Cloud
- ✅ Procesamiento de imágenes en GPU con CUDA/PyCUDA
- ✅ App móvil nativa con Flutter
- ✅ Autenticación segura con Firebase
- ✅ Almacenamiento escalable con Supabase
- ✅ Containerización con Docker

---

## 🏗️ Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────┐
│                     MOBILE APP                            │
│                   (Flutter 3.10)                          │
│  - Camera & Gallery                                       │
│  - Real-time Filter Preview                               │
│  - Social Feed                                            │
│  - User Profiles                                          │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTP/REST
                     ↓
┌──────────────────────────────────────────────────────────┐
│                   API GATEWAY                             │
│              (Spring Cloud Gateway)                       │
│  Port: 8080                                               │
│  - Request Routing                                        │
│  - CORS Configuration                                     │
│  - Load Balancing                                         │
└────────┬────────────────┬─────────────────┬──────────────┘
         │                │                 │
         ↓                ↓                 ↓
┌─────────────────┐ ┌─────────────────┐ ┌──────────────────┐
│  AUTH SERVICE   │ │  POST SERVICE   │ │  CUDA BACKEND    │
│  (Spring Boot)  │ │  (Spring Boot)  │ │  (Python+CUDA)   │
│  Port: 8082     │ │  Port: 8081     │ │  Port: 5000      │
│                 │ │                 │ │                  │
│ • Login/Register│ │ • Posts CRUD    │ │ • 7 GPU Filters  │
│ • User Mgmt     │ │ • Likes         │ │ • PyCUDA Kernels │
│ • Follow System │ │ • Comments      │ │ • Image Process  │
│ • Avatar Upload │ │ • Feed          │ │ • FastAPI        │
└────────┬────────┘ └────────┬────────┘ └────────┬─────────┘
         │                   │                    │
         ↓                   ↓                    ↓
┌─────────────────┐ ┌──────────────────┐ ┌───────────────┐
│    FIREBASE     │ │     SUPABASE     │ │  NVIDIA GPU   │
│                 │ │                  │ │               │
│ • Firestore     │ │ • PostgreSQL     │ │ • CUDA 12.x   │
│ • Auth          │ │ • R2DBC          │ │ • Parallel    │
│ • Storage       │ │ • Object Storage │ │   Processing  │
└─────────────────┘ └──────────────────┘ └───────────────┘
```

---

## 🚀 Stack Tecnológico

### Backend Microservices
| Componente | Tecnología | Puerto |
|-----------|------------|--------|
| **API Gateway** | Spring Cloud Gateway 2023.0.0 | 8080 |
| **Auth Service** | Spring Boot 3.2.0 + WebFlux | 8082 |
| **Post Service** | Spring Boot 3.2.0 + R2DBC | 8081 |
| **CUDA Backend** | Python 3.10 + FastAPI 0.122 | 5000 |

### Databases & Storage
- **Firebase**: Firestore (NoSQL), Authentication, Storage
- **Supabase**: PostgreSQL (R2DBC), Object Storage

### Mobile App
- **Flutter**: 3.10.1
- **Dart**: 3.10.1
- **Packages**: image_picker, camera, http, shared_preferences

### Infrastructure
- **Java**: 21 (Eclipse Temurin)
- **Maven**: 3.9+
- **Docker**: Multi-stage builds
- **CUDA**: 12.x + NVIDIA GPU
- **PyCUDA**: 2025.1.2

---

## 📁 Estructura del Proyecto

```
UPSGlam-2.0/
├── backend-java/                      # Java Microservices
│   ├── api-gateway/                   # Spring Cloud Gateway
│   ├── auth-service/                  # Authentication + Firebase
│   ├── post-service/                  # Posts, Likes, Comments + Supabase
│   ├── cuda-lab-back/                 # Python CUDA Processing
│   ├── pycuda-mock/                   # Mock service for testing
│   ├── docker-compose.yml             # Orchestration
│   ├── .env                           # Environment variables (gitignored)
│   └── README-TECHNICAL.md            # Backend documentation
│
├── cuda-service/                      # Alternative CUDA service
│   ├── app/
│   ├── filters/
│   └── README-DETAILED.md
│
├── mobile_app/                        # Flutter Mobile App
│   ├── lib/
│   │   ├── screens/                   # UI Screens
│   │   ├── services/                  # API Clients
│   │   ├── models/                    # Data Models
│   │   └── main.dart                  # Entry Point
│   ├── android/
│   ├── ios/
│   └── README-DETAILED.md             # Mobile app docs
│
├── filter_preview_app/                # Local filter preview package
│   └── lib/
│
├── docs/                              # General documentation
│   ├── README-BACKEND-GUIDE.MD
│   ├── README-FRONTEND-GUIDE.MD
│   ├── README-PYCUDA-GUIDE.MD
│   └── README-SUPABASE-GUIDE.MD
│
└── infra/                             # Infrastructure configs
```

---

## ✨ Características Principales

### 🔐 Autenticación
- Login/Register con Firebase Authentication
- JWT tokens para autorización
- Gestión de perfiles de usuario
- Upload de avatares a Firebase Storage

### 📸 Creación de Posts
- Captura de foto con cámara
- Selección desde galería
- Preview en vivo con filtros locales (Dart)
- Aplicación de filtros GPU (CUDA)
- Upload a Supabase Storage
- Descripción de post

### 🎨 Filtros GPU (CUDA)
1. **Gaussian Blur** - Desenfoque gaussiano
2. **Box Blur** - Desenfoque de caja
3. **Prewitt** - Detección de bordes Prewitt
4. **Laplacian** - Detección de bordes Laplacian
5. **UPS Logo** - Overlay del logo UPS
6. **Boomerang** - Efecto boomerang
7. **CR7** - Máscara CR7

### 📱 Feed Social
- Timeline con posts de usuarios seguidos
- Pull-to-refresh
- Scroll infinito
- Likes en tiempo real
- Sistema de comentarios
- Navegación a perfiles

### 👤 Perfiles de Usuario
- Ver perfil propio y de otros
- Grid de posts del usuario
- Contador de posts/seguidores/seguidos
- Follow/Unfollow
- Edición de perfil

---

## 🚀 Quick Start

### 1. Prerrequisitos

```bash
# Backend
java -version          # Java 21
mvn -version           # Maven 3.9+
docker --version       # Docker
nvidia-smi             # NVIDIA GPU

# Mobile
flutter --version      # Flutter 3.10+
```

### 2. Configurar Backend

```bash
cd backend-java

# Copiar plantillas de configuración
cp .env.example .env
cp docker-compose.yml.example docker-compose.yml

# Editar con tus credenciales
notepad .env

# Descargar firebase-credentials.json desde Firebase Console
# y guardarlo en backend-java/firebase-credentials.json

# Iniciar todos los servicios
docker-compose up -d --build

# Verificar logs
docker-compose logs -f
```

### 3. Configurar Mobile App

```bash
cd mobile_app

# Instalar dependencias
flutter pub get

# Configurar URL del backend
# Editar lib/services/auth_service.dart
# baseUrl = 'http://10.0.2.2:8080/api'  # Android Emulator
# baseUrl = 'http://localhost:8080/api' # iOS Simulator

# Ejecutar app
flutter run
```

### 4. Verificar Servicios

```bash
# Health checks
curl http://localhost:8080/health  # API Gateway
curl http://localhost:8082/api/auth/health  # Auth
curl http://localhost:8081/health  # Posts
curl http://localhost:5000/health  # CUDA

# Test completo
cd backend-java/api-gateway
.\test-gateway.ps1
```

---

## 📡 API Endpoints

### Authentication (`/api/auth/*`)
```bash
POST /api/auth/login              # Login
POST /api/auth/register           # Register
GET  /api/auth/user/{userId}      # Get user
POST /api/auth/user/{userId}/avatar  # Upload avatar
POST /api/auth/follows            # Follow user
DELETE /api/auth/follows/{followingId}  # Unfollow
```

### Posts (`/api/posts/*`)
```bash
GET    /api/posts                 # All posts
POST   /api/posts                 # Create post
GET    /api/posts/{id}            # Get post
DELETE /api/posts/{id}            # Delete post
POST   /api/posts/{id}/like       # Like post
DELETE /api/posts/{id}/like       # Unlike post
GET    /api/feed                  # Personalized feed
```

### Comments (`/api/posts/{postId}/comments`)
```bash
GET  /api/posts/{postId}/comments     # Get comments
POST /api/posts/{postId}/comments     # Add comment
```

### Filters (`/api/filters/*`)
```bash
POST /api/filters/apply           # Apply CUDA filter
FormData:
  - image: File
  - filter_name: string
```

Ver documentación completa: [API-ROUTES.md](./backend-java/api-gateway/API-ROUTES.md)

---

## 🔧 Configuración

### Variables de Entorno (`.env`)

```properties
# Firebase
FIREBASE_PROJECT_ID=tu-proyecto-firebase
FIREBASE_API_KEY=AIzaSy...
FIREBASE_STORAGE_BUCKET=tu-proyecto.appspot.com

# Supabase
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_KEY=eyJhbGc...
SUPABASE_STORAGE_BUCKET=images
```

### Archivos Sensibles (Gitignored)

```
backend-java/.env
backend-java/docker-compose.yml
backend-java/firebase-credentials.json
backend-java/auth-service/src/main/resources/application-docker.yml
backend-java/post-service/src/main/resources/application-docker.yml
```

**Plantillas disponibles**: `.env.example`, `docker-compose.yml.example`, `application-docker.yml.example`

---

## 🧪 Testing

### Backend Tests

```bash
# API Gateway
cd backend-java/api-gateway
.\test-gateway.ps1

# Auth Service
cd backend-java/auth-service
.\test-auth.ps1
.\test-follows-complete.ps1

# Post Service
cd backend-java/post-service
.\test-api.ps1
.\test-endpoints.ps1

# CUDA Backend
cd backend-java/cuda-lab-back
python test_curl.py
```

### Mobile Tests

```bash
cd mobile_app

# Unit tests
flutter test

# Integration tests
flutter test integration_test/
```

---

## 📊 Performance

### CUDA Processing Benchmarks
| Filter | Resolution | CPU Time | GPU Time | Speedup |
|--------|-----------|----------|----------|---------|
| Gaussian | 1920x1080 | 450ms | 12ms | 37.5x |
| Box Blur | 1920x1080 | 380ms | 9ms | 42.2x |
| Prewitt | 1920x1080 | 520ms | 15ms | 34.7x |
| Laplacian | 1920x1080 | 510ms | 14ms | 36.4x |

### Mobile App
- ✅ 60fps UI rendering
- ✅ < 2s image upload
- ✅ Real-time filter preview
- ✅ Optimized image caching

---

## 🐛 Troubleshooting

### Backend no inicia

```bash
# Verificar Docker está corriendo
docker ps

# Verificar puertos disponibles
netstat -ano | findstr :8080

# Reconstruir desde cero
cd backend-java
docker-compose down -v
docker-compose up --build
```

### Mobile app no conecta

```dart
// Android Emulator
baseUrl = 'http://10.0.2.2:8080/api';

// iOS Simulator
baseUrl = 'http://localhost:8080/api';

// Dispositivo físico (misma WiFi)
baseUrl = 'http://192.168.1.100:8080/api';
```

### CUDA no disponible

```bash
# Verificar driver
nvidia-smi

# Verificar Docker tiene acceso a GPU
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Reinstalar NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

---

## 📚 Documentación Detallada

### Backend
- [Backend Java - Arquitectura General](./backend-java/README-TECHNICAL.md)
- [API Gateway](./backend-java/api-gateway/README-DETAILED.md)
- [Auth Service](./backend-java/auth-service/README-DETAILED.md)
- [Post Service](./backend-java/post-service/README-DETAILED.md)
- [CUDA Backend](./backend-java/cuda-lab-back/README-DETAILED.md)

### Mobile
- [Mobile App - Flutter](./mobile_app/README-DETAILED.md)

### Otros
- [CUDA Service Alternativo](./cuda-service/README-DETAILED.md)
- [Guías Generales](./docs/)

---

## 🛠️ Tecnologías Utilizadas

### Backend
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.2.0-brightgreen)
![Java](https://img.shields.io/badge/Java-21-orange)
![Docker](https://img.shields.io/badge/Docker-Latest-blue)

### Mobile
![Flutter](https://img.shields.io/badge/Flutter-3.10-blue)
![Dart](https://img.shields.io/badge/Dart-3.10-blue)

### AI/ML
![CUDA](https://img.shields.io/badge/CUDA-12.x-green)
![Python](https://img.shields.io/badge/Python-3.10-yellow)

### Cloud
![Firebase](https://img.shields.io/badge/Firebase-Latest-orange)
![Supabase](https://img.shields.io/badge/Supabase-Latest-green)

---

## 👥 Equipo de Desarrollo

**UPSGlam Development Team**  
Universidad Politécnica Salesiana  
Quito, Ecuador

### Integrantes
- Desarrollo Backend (Java/Spring)
- Desarrollo Mobile (Flutter)
- Desarrollo CUDA (Python/PyCUDA)
- Infraestructura (Docker/Cloud)

---

## 📄 Licencia

Este proyecto es privado y confidencial.  
**© 2025 Universidad Politécnica Salesiana**  
Todos los derechos reservados.

---

## 🎓 Contexto Académico

Proyecto desarrollado como parte del programa de Ingeniería de Software de la Universidad Politécnica Salesiana. Implementa conceptos avanzados de:

- ✅ Arquitectura de Microservicios
- ✅ Computación en GPU con CUDA
- ✅ Desarrollo Móvil Multiplataforma
- ✅ Cloud Computing y Servicios Serverless
- ✅ CI/CD y Containerización
- ✅ APIs RESTful y Reactive Programming
- ✅ Bases de Datos Relacionales y NoSQL

---

## 📞 Contacto y Soporte

Para preguntas sobre el proyecto:
- **Universidad**: Universidad Politécnica Salesiana
- **Campus**: Quito, Ecuador
- **Año**: 2025

---

## 🔗 Enlaces Útiles

- [Spring Boot Documentation](https://spring.io/projects/spring-boot)
- [Flutter Documentation](https://docs.flutter.dev/)
- [PyCUDA Documentation](https://documen.tician.de/pycuda/)
- [Firebase Documentation](https://firebase.google.com/docs)
- [Supabase Documentation](https://supabase.com/docs)
- [Docker Documentation](https://docs.docker.com/)

---

**⭐ Si este proyecto te fue útil, déjanos una estrella!**
