# 📱 UPSGlam 2.0

[![Flutter](https://img.shields.io/badge/Flutter-3.10+-blue.svg)](https://flutter.dev/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.2.0-green.svg)](https://spring.io/projects/spring-boot)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore-orange.svg)](https://firebase.google.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)

Red social universitaria con procesamiento de imágenes mediante CUDA, desarrollada con arquitectura de microservicios.

## 🚀 Descargar Aplicación

### 📥 APK Android (Última Versión)

**[⬇️ Descargar UPSGlam 2.0 APK](https://drive.google.com/file/d/1_wT09RQ2KxfvcuT_qxe8eaY2Hy4R3wWb/view?usp=sharing)**

- **Versión:** 2.0.0
- **Tamaño:** ~64 MB
- **Android:** 6.0 o superior
- **Fecha:** Diciembre 12, 2025

---

## 🌟 Características Principales

### Para Usuarios
- 🔐 **Autenticación Segura** con Firebase (Email/Password)
- 📸 **Creación de Posts** con imágenes y audio opcional
- 🎨 **Filtros CUDA** - Procesamiento de imágenes con GPU (marca de agua UPS, blur, edge detection, etc.)
- ❤️ **Likes y Comentarios** en tiempo real
- 🔔 **Notificaciones Push** - Likes, comentarios y nuevos seguidores
- 👥 **Sistema de Seguidos** - Sigue a otros usuarios
- 📰 **Feed Personalizado** con scroll infinito
- 🚫 **Censura de Contenido** - Bloqueo automático de palabras prohibidas
- 🔊 **Reproductor de Audio** integrado en posts
- 🔍 **Búsqueda de Usuarios**
- 👤 **Perfiles Públicos** con contador de posts/seguidores

### Para Desarrolladores
- 🏗️ **Arquitectura de Microservicios** con Spring Boot WebFlux
- 🐳 **Docker Compose** para despliegue simple
- 🔥 **Firebase/Firestore** para autenticación y notificaciones
- 🐘 **Supabase PostgreSQL** para posts y datos relacionales
- ⚡ **Programación Reactiva** con R2DBC y WebFlux
- 🎮 **Procesamiento CUDA** con PyCUDA
- 🌐 **API Gateway** centralizado con Spring Cloud Gateway
- 📊 **Health Checks** y monitoreo con Spring Actuator

---

## 📋 Tabla de Contenidos

- [Arquitectura](#-arquitectura)
- [Tecnologías](#-tecnologías)
- [Requisitos](#-requisitos)
- [Instalación Rápida](#-instalación-rápida)
- [Despliegue Detallado](#-despliegue-detallado)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [APIs y Endpoints](#-apis-y-endpoints)
- [Configuración](#-configuración)
- [Capturas de Pantalla](#-capturas-de-pantalla)
- [Solución de Problemas](#-solución-de-problemas)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

---

## 🏗️ Arquitectura

### Diagrama de Sistema

```
┌─────────────────────────────────────────────────────────┐
│              Mobile App (Flutter)                       │
│           iOS / Android / Web                           │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/REST (Port 8080)
                   ▼
┌─────────────────────────────────────────────────────────┐
│              API Gateway (Spring Cloud)                 │
│         Routing • Auth • Load Balancing                 │
└──────┬──────────┬──────────────┬───────────────────────┘
       │          │              │
       ▼          ▼              ▼
   ┌────────┐ ┌──────────┐ ┌─────────────┐
   │Auth    │ │Post      │ │CUDA         │
   │Service │ │Service   │ │Backend      │
   │:8082   │ │:8081     │ │:5000        │
   └───┬────┘ └────┬─────┘ └─────────────┘
       │           │
       ▼           ▼
   ┌──────────────────────────────────┐
   │     Firebase Firestore           │
   │  Authentication • Notifications  │
   └──────────────────────────────────┘
       │           
       ▼           
   ┌──────────────────────────────────┐
   │    Supabase PostgreSQL           │
   │ Posts • Likes • Comments         │
   └──────────────────────────────────┘
```

### Microservicios

| Servicio | Puerto | Tecnología | Función |
|----------|--------|------------|---------|
| **API Gateway** | 8080 | Spring Cloud Gateway | Enrutamiento, autenticación |
| **Auth Service** | 8082 | Spring Boot + Firebase | Usuarios, autenticación, seguidos |
| **Post Service** | 8081 | Spring Boot + Supabase | Posts, likes, comentarios, notificaciones |
| **CUDA Backend** | 5000 | Python + PyCUDA | Procesamiento de imágenes con GPU |

---

## 💻 Tecnologías

### Frontend (Mobile App)
- **Flutter** 3.10+ (Dart)
- **Firebase Auth** - Autenticación
- **HTTP** - Cliente REST
- **Shared Preferences** - Almacenamiento local
- **Audioplayers** - Reproducción de audio
- **Visibility Detector** - Detección de visibilidad

### Backend
- **Spring Boot** 3.2.0 - Framework Java
- **Spring WebFlux** - Programación reactiva
- **Spring Cloud Gateway** - API Gateway
- **R2DBC** - Acceso reactivo a PostgreSQL
- **Firebase Admin SDK** - Autenticación y Firestore
- **Maven** - Gestión de dependencias

### Base de Datos
- **Firebase Firestore** - Usuarios y notificaciones
- **Supabase PostgreSQL** - Posts, likes, comentarios
- **Supabase Storage** - Almacenamiento de imágenes y audio

### Procesamiento de Imágenes
- **Python** 3.8+
- **PyCUDA** - Procesamiento paralelo en GPU
- **NumPy** - Operaciones matriciales
- **Pillow** - Manipulación de imágenes
- **Flask** - API REST para filtros

### DevOps
- **Docker** & **Docker Compose** - Contenedorización
- **Git** - Control de versiones
- **PowerShell** - Scripts de automatización

---

## 📦 Requisitos

### Para Usuarios (Instalar APK)
- ✅ Dispositivo Android 6.0 o superior
- ✅ ~64 MB de espacio libre
- ✅ Conexión a internet

### Para Desarrolladores (Ejecutar Backend)
- ✅ Docker Desktop 20.10+
- ✅ Java JDK 21
- ✅ Maven 3.9+
- ✅ Git
- ⚠️ NVIDIA GPU (opcional, solo para filtros CUDA reales)

### Para Desarrolladores (Ejecutar App Móvil)
- ✅ Flutter SDK 3.10+
- ✅ Android Studio / VS Code
- ✅ Android SDK / Xcode (según plataforma)

---

## ⚡ Instalación Rápida

### Opción 1: Solo usar la App (Usuarios Finales)

1. **Descargar APK**
   ```
   https://drive.google.com/file/d/1_wT09RQ2KxfvcuT_qxe8eaY2Hy4R3wWb/view?usp=sharing
   ```

2. **Instalar en Android**
   - Permitir instalación de fuentes desconocidas
   - Abrir el APK descargado
   - Seguir instrucciones de instalación

3. **Crear Cuenta**
   - Abrir UPSGlam
   - Registrarse con email y contraseña
   - ¡Listo para usar!

### Opción 2: Despliegue Completo (Desarrolladores)

#### Paso 1: Clonar Repositorio
```bash
git clone https://github.com/tu-usuario/UPSGlam-2.0.git
cd UPSGlam-2.0
```

#### Paso 2: Configurar Firebase

1. Crear proyecto en [Firebase Console](https://console.firebase.google.com/)
2. Descargar `firebase-credentials.json`
3. Copiar a `backend-java/auth-service/src/main/resources/`
4. Descargar `google-services.json` (Android)
5. Copiar a `mobile_app/android/app/`

#### Paso 3: Configurar Supabase

1. Crear proyecto en [Supabase](https://supabase.com/)
2. Editar `backend-java/post-service/src/main/resources/application-docker.yml`:
   ```yaml
   supabase:
     url: https://tu-proyecto.supabase.co
     service-role-key: tu-service-role-key
   ```

#### Paso 4: Levantar Backend
```bash
cd backend-java
docker-compose up -d
```

#### Paso 5: Configurar IP en App
```dart
// mobile_app/lib/config/api_config.dart
static const String _baseUrl = 'http://TU_IP:8080';
```

#### Paso 6: Ejecutar App
```bash
cd mobile_app
flutter pub get
flutter run
```

---

## 📖 Despliegue Detallado

Para guía completa de despliegue, configuración y uso, ver:

📘 **[DEPLOYMENT-GUIDE.md](DEPLOYMENT-GUIDE.md)** - Documentación técnica completa

Incluye:
- Configuración paso a paso de Firebase y Supabase
- Scripts SQL para crear tablas
- Configuración de Docker
- Construcción de APKs
- Solución de problemas comunes
- Documentación completa de APIs

---

## 📁 Estructura del Proyecto

```
UPSGlam-2.0/
├── backend-java/                    # Backend (Microservicios Java)
│   ├── api-gateway/                 # API Gateway (Puerto 8080)
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── pom.xml
│   ├── auth-service/                # Servicio de Autenticación (Puerto 8082)
│   │   ├── src/
│   │   │   └── main/resources/
│   │   │       └── firebase-credentials.json   # ⚠️ Configurar
│   │   ├── Dockerfile
│   │   └── pom.xml
│   ├── post-service/                # Servicio de Posts (Puerto 8081)
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── pom.xml
│   ├── cuda-lab-back/               # Procesamiento CUDA (Puerto 5000)
│   │   ├── app.py
│   │   ├── filters/
│   │   └── requirements.txt
│   └── docker-compose.yml           # Orquestación de servicios
│
├── mobile_app/                      # Aplicación Flutter
│   ├── lib/
│   │   ├── main.dart
│   │   ├── config/
│   │   │   └── api_config.dart      # ⚠️ Configurar IP
│   │   ├── screens/
│   │   ├── services/
│   │   └── models/
│   ├── android/
│   │   └── app/
│   │       └── google-services.json # ⚠️ Configurar
│   ├── build/
│   │   └── app/outputs/flutter-apk/
│   │       └── app-release.apk      # APK generado
│   ├── pubspec.yaml
│   ├── install-apk.ps1              # Script de instalación
│   └── README.md
│
├── docs/                            # Documentación adicional
│   ├── README-BACKEND-GUIDE.MD
│   ├── README-FRONTEND-GUIDE.MD
│   └── README-PYCUDA-GUIDE.MD
│
├── DEPLOYMENT-GUIDE.md              # Guía completa de despliegue
└── README.md                        # Este archivo
```

---

## 🔌 APIs y Endpoints

### API Gateway (http://localhost:8080)

Todos los requests pasan por el gateway en el puerto **8080**.

#### Autenticación

```http
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
GET  /api/auth/users/{userId}
POST /api/auth/follow/{userId}
GET  /api/auth/search?query={username}
```

#### Posts

```http
POST /api/posts
GET  /api/feed?page=0&size=20
GET  /api/posts/{postId}
POST /api/posts/{postId}/like
GET  /api/posts/{postId}/comments
POST /api/posts/{postId}/comments
POST /api/images/upload
```

#### Notificaciones

```http
GET   /api/notifications/me
POST  /api/notifications
PATCH /api/notifications/{id}/read
```

#### Filtros CUDA

```http
POST /api/filters/{filterName}
Content-Type: image/jpeg
Body: [Binary Image Data]
```

**Filtros disponibles:**
- `ups_logo` - Marca de agua UPS
- `blox_blur` - Desenfoque
- `edge_detection` - Detección de bordes
- `sharpen` - Afilado
- `emboss` - Relieve
- `grayscale` - Escala de grises
- `sepia` - Efecto sepia
- `invert` - Invertir colores

### Headers Requeridos

```http
Authorization: Bearer {firebase-jwt-token}
X-User-Id: {userId}
X-Username: {username}
Content-Type: application/json
```

---

## ⚙️ Configuración

### Variables de Entorno (Backend)

#### Auth Service
```yaml
# application-docker.yml
firebase:
  credentials:
    path: /app/firebase-credentials.json
  firestore:
    database: db-auth
```

#### Post Service
```yaml
# application-docker.yml
supabase:
  url: ${SUPABASE_URL}
  service-role-key: ${SUPABASE_KEY}
  storage:
    bucket-name: post-images

firebase:
  credentials:
    path: /app/firebase-credentials.json
```

### Configuración de Red

Por defecto, el backend escucha en todas las interfaces (`0.0.0.0`).

Para acceder desde la app móvil:
1. Encontrar IP de tu máquina: `ipconfig` (Windows) o `ifconfig` (Linux/Mac)
2. Actualizar en `mobile_app/lib/config/api_config.dart`:
   ```dart
   static const String _baseUrl = 'http://192.168.X.X:8080';
   ```

### Puertos Utilizados

| Puerto | Servicio |
|--------|----------|
| 8080 | API Gateway |
| 8081 | Post Service |
| 8082 | Auth Service |
| 5000 | CUDA Backend |

---

## 📸 Capturas de Pantalla

### Pantallas Principales

| Login | Feed | Crear Post |
|-------|------|------------|
| 🔐 Autenticación con Firebase | 📰 Feed infinito con posts | 📸 Captura y filtros |

| Perfil | Notificaciones | Comentarios |
|--------|----------------|-------------|
| 👤 Perfil con seguidores | 🔔 Notificaciones en tiempo real | 💬 Comentarios en posts |

---

## 🐛 Solución de Problemas

### Problema: "Connection Timeout" en la App

**Solución:**
1. Verificar que el backend está corriendo:
   ```bash
   docker ps
   ```
2. Verificar la IP configurada en `api_config.dart`
3. Verificar que el firewall permite el puerto 8080

### Problema: Servicios "unhealthy" en Docker

**Solución:**
```bash
# Ver logs del servicio
docker logs upsglam-auth-service

# Reconstruir y reiniciar
docker-compose down
docker-compose build
docker-compose up -d
```

### Problema: Firebase Authentication Error

**Solución:**
1. Verificar que `firebase-credentials.json` está en la ruta correcta
2. Verificar que el proyecto de Firebase tiene Authentication habilitado
3. Reconstruir el contenedor del auth-service

### Problema: Posts no se muestran en el Feed

**Solución:**
1. Verificar conexión a Supabase en los logs del post-service
2. Verificar que las tablas existen en Supabase
3. Crear un post de prueba desde la app

Para más soluciones, ver **[DEPLOYMENT-GUIDE.md](DEPLOYMENT-GUIDE.md#-solución-de-problemas)**

---

## 🔒 Seguridad

### Autenticación
- JWT tokens de Firebase para todas las peticiones
- Validación en API Gateway
- Headers X-User-Id y X-Username obligatorios

### Censura de Contenido
Lista de palabras prohibidas al crear posts:
- messi, barcelona, visca barca, barça
- hitler, nazi
- puto, pendejo

### Buenas Prácticas
- ⚠️ Nunca commitear `firebase-credentials.json`
- ⚠️ Nunca commitear `google-services.json`
- ⚠️ Usar variables de entorno para secrets en producción
- ⚠️ Configurar CORS apropiadamente
- ⚠️ Usar HTTPS en producción

---

## 📊 Base de Datos

### Firebase Firestore

**Database: db-auth**
- Colección `users` - Información de usuarios

**Database: (default)**
- Colección `notifications` - Notificaciones de usuarios

### Supabase PostgreSQL

**Tablas:**
- `posts` - Publicaciones de usuarios
- `likes` - Likes en posts
- `comments` - Comentarios en posts

**Storage:**
- Bucket `post-images` - Imágenes de posts
- Bucket `post-audios` - Audios opcionales

---

## 🚢 Scripts de Despliegue

### Backend

```bash
# Iniciar todos los servicios
cd backend-java
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down

# Reconstruir servicio específico
docker-compose build auth-service
docker-compose up -d auth-service
```

### Mobile App

```bash
# Construir APK
cd mobile_app
flutter build apk --release

# Instalar en dispositivo conectado
flutter run --release

# O usar script de instalación
.\install-apk.ps1
```

---

## 🤝 Contribuir

### Flujo de Trabajo

1. Fork el proyecto
2. Crear rama de feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

### Estándares de Código

- **Java**: Seguir convenciones de Spring Boot
- **Dart/Flutter**: Seguir guías oficiales de Flutter
- **Python**: PEP 8
- **Commits**: Mensajes descriptivos en inglés

---

## 📝 Roadmap

### Versión Actual (2.0.0)
- ✅ Arquitectura de microservicios
- ✅ Sistema de notificaciones
- ✅ Filtros CUDA
- ✅ Censura de contenido
- ✅ Sistema de likes y comentarios
- ✅ Sistema de seguidos

### Futuras Mejoras
- [ ] Chat en tiempo real
- [ ] Stories temporales (24h)
- [ ] Reels/Videos cortos
- [ ] Mensajería directa
- [ ] Modo oscuro
- [ ] Login con Google/Facebook
- [ ] Recuperación de contraseña
- [ ] Verificación de email
- [ ] Analytics y estadísticas
- [ ] Moderación automática con IA

---

## 👥 Equipo

**Universidad Politécnica Salesiana**
- Proyecto de Red Social con Procesamiento CUDA
- Arquitectura de Microservicios

---

## 📄 Licencia

Este proyecto es desarrollado con fines educativos para la Universidad Politécnica Salesiana.

---

## 📞 Soporte

### Reportar Bugs
- Crear issue en GitHub con:
  - Descripción del problema
  - Pasos para reproducir
  - Logs relevantes
  - Capturas de pantalla

### Documentación Adicional
- 📘 [DEPLOYMENT-GUIDE.md](DEPLOYMENT-GUIDE.md) - Guía completa de despliegue
- 📗 [docs/README-BACKEND-GUIDE.MD](docs/README-BACKEND-GUIDE.MD) - Backend
- 📙 [docs/README-FRONTEND-GUIDE.MD](docs/README-FRONTEND-GUIDE.MD) - Frontend
- 📕 [docs/README-PYCUDA-GUIDE.MD](docs/README-PYCUDA-GUIDE.MD) - CUDA

---

## 🌟 Agradecimientos

- Firebase por el servicio de autenticación
- Supabase por la base de datos y storage
- Spring Boot por el framework de microservicios
- Flutter por el framework de desarrollo móvil
- NVIDIA por CUDA y PyCUDA

---

<div align="center">

**Hecho con ❤️ por el equipo UPSGlam**

[⬆ Volver arriba](#-upsglam-20)

</div>