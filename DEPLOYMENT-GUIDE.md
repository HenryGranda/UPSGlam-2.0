# UPSGlam 2.0 - Guía de Despliegue y Uso

## 📋 Tabla de Contenidos

- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Requisitos Previos](#requisitos-previos)
- [Configuración del Entorno](#configuración-del-entorno)
- [Despliegue de Servicios Backend](#despliegue-de-servicios-backend)
- [Despliegue de la Aplicación Móvil](#despliegue-de-la-aplicación-móvil)
- [Monitoreo y Salud de Servicios](#monitoreo-y-salud-de-servicios)
- [Solución de Problemas](#solución-de-problemas)
- [APIs y Endpoints](#apis-y-endpoints)

---

## 🏗️ Arquitectura del Sistema

### Visión General

UPSGlam es una aplicación de red social con procesamiento de imágenes mediante CUDA, construida con una arquitectura de microservicios.

```
┌─────────────────────────────────────────────────────────────┐
│                     Mobile App (Flutter)                     │
│              iOS / Android / Web / Desktop                   │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTP/REST
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway (8080)                        │
│              Spring Cloud Gateway / WebFlux                  │
│          Routing, Load Balancing, Authentication            │
└─────┬──────────┬──────────┬────────────────────────────────┘
      │          │          │
      ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌─────────────────┐
│Auth Svc  │ │Post Svc  │ │CUDA Backend     │
│  (8082)  │ │  (8081)  │ │    (5000)       │
└────┬─────┘ └────┬─────┘ └────────┬────────┘
     │            │                 │
     ▼            ▼                 ▼
┌─────────────────────────────────────────┐
│         Firebase / Firestore            │
│  Authentication, Users, Notifications   │
└─────────────────────────────────────────┘
     │            
     ▼            
┌─────────────────────────────────────────┐
│         Supabase PostgreSQL             │
│      Posts, Likes, Comments, Follows    │
└─────────────────────────────────────────┘
```

### Componentes Principales

#### 1. **API Gateway** (Puerto 8080)
- **Tecnología**: Spring Cloud Gateway con WebFlux (Reactivo)
- **Función**: 
  - Punto de entrada único para todas las peticiones
  - Enrutamiento dinámico a microservicios
  - Validación de tokens JWT
  - Headers forwarding (X-User-Id, X-Username)
- **Rutas**:
  - `/api/auth/*` → Auth Service
  - `/api/posts/*`, `/api/feed`, `/api/images/*` → Post Service
  - `/api/filters/*` → CUDA Backend

#### 2. **Auth Service** (Puerto 8082)
- **Tecnología**: Spring Boot WebFlux + Firebase Admin SDK
- **Base de Datos**: Firestore (`db-auth`)
- **Funciones**:
  - Registro y login de usuarios
  - Validación de tokens JWT con Firebase
  - Gestión de perfiles de usuario
  - Seguir/dejar de seguir usuarios
  - Búsqueda de usuarios
- **Endpoints Principales**:
  - `POST /api/auth/register` - Registro
  - `POST /api/auth/login` - Inicio de sesión
  - `GET /api/auth/me` - Perfil actual
  - `GET /api/auth/users/{userId}` - Perfil de usuario
  - `POST /api/auth/follow/{userId}` - Seguir usuario

#### 3. **Post Service** (Puerto 8081)
- **Tecnología**: Spring Boot WebFlux + R2DBC
- **Bases de Datos**: 
  - Supabase PostgreSQL (posts, likes, comments)
  - Firestore (notifications)
- **Funciones**:
  - Creación y gestión de posts
  - Sistema de likes
  - Sistema de comentarios
  - Feed personalizado
  - Notificaciones (likes, comments, follows)
  - Almacenamiento de imágenes en Supabase Storage
- **Endpoints Principales**:
  - `POST /api/posts` - Crear post
  - `GET /api/feed` - Feed paginado
  - `POST /api/posts/{postId}/like` - Dar like
  - `POST /api/posts/{postId}/comments` - Comentar
  - `GET /api/notifications/me` - Obtener notificaciones

#### 4. **CUDA Backend** (Puerto 5000)
- **Tecnología**: Python + PyCUDA + Flask
- **Función**: Procesamiento de imágenes con GPU
- **Filtros Disponibles**:
  - `ups_logo` - Marca de agua UPS
  - `blox_blur` - Desenfoque
  - `edge_detection` - Detección de bordes
  - `sharpen` - Afilado
  - `emboss` - Relieve
  - Y más...
- **Endpoint**: `POST /filters/{filterName}`

#### 5. **Mobile App** (Flutter)
- **Tecnología**: Flutter 3.10+ (Dart)
- **Plataformas**: iOS, Android, Web, Windows, macOS, Linux
- **Características**:
  - Autenticación con Firebase
  - Feed de posts con scroll infinito
  - Creación de posts con filtros CUDA
  - Likes y comentarios en tiempo real
  - Sistema de notificaciones
  - Perfiles de usuario
  - Seguir/dejar de seguir
  - **Censura de contenido** (palabras prohibidas)
  - Reproductor de audio integrado

---

## 🔧 Requisitos Previos

### Software Necesario

1. **Docker Desktop** (Windows/Mac/Linux)
   - Versión: 20.10+
   - Docker Compose: v2.0+

2. **Java Development Kit**
   - Versión: JDK 21 (Eclipse Temurin recomendado)
   - Maven 3.9+

3. **Flutter SDK**
   - Versión: 3.10.0+
   - Dart: 3.10+

4. **NVIDIA Docker** (solo para CUDA Backend)
   - NVIDIA GPU con soporte CUDA
   - NVIDIA Docker Runtime

5. **Git**
   - Para clonar el repositorio

### Servicios Externos

1. **Firebase**
   - Proyecto creado en Firebase Console
   - Authentication habilitado (Email/Password)
   - Firestore Database creado
   - Archivo `firebase-credentials.json` descargado

2. **Supabase**
   - Proyecto creado en Supabase
   - PostgreSQL Database activo
   - Storage Bucket configurado
   - Service Role Key obtenida

---

## ⚙️ Configuración del Entorno

### 1. Firebase Setup

#### Crear Proyecto Firebase

1. Ir a [Firebase Console](https://console.firebase.google.com/)
2. Crear nuevo proyecto: `upsglam-8c88f` (o tu nombre)
3. Habilitar **Authentication** → Email/Password
4. Crear **Firestore Database**:
   - Database ID: `db-auth`
   - Ubicación: us-central (o tu preferencia)
5. Crear segunda database (para notificaciones):
   - Database ID: `(default)`

#### Obtener Credenciales

1. Ir a **Project Settings** → **Service Accounts**
2. Click en **Generate New Private Key**
3. Guardar archivo como `firebase-credentials.json`
4. Copiar a: `backend-java/auth-service/src/main/resources/firebase-credentials.json`

#### Configurar App Móvil

**Android:**
```bash
# Descargar google-services.json desde Firebase Console
# Ubicación: mobile_app/android/app/google-services.json
```

**iOS:**
```bash
# Descargar GoogleService-Info.plist desde Firebase Console
# Ubicación: mobile_app/ios/Runner/GoogleService-Info.plist
```

### 2. Supabase Setup

#### Crear Proyecto

1. Ir a [Supabase Dashboard](https://supabase.com/dashboard)
2. Crear nuevo proyecto
3. Anotar:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **Service Role Key**: `eyJhbGc...`
   - **Anon Public Key**: `eyJhbGc...`

#### Crear Tablas

Ejecutar en SQL Editor:

```sql
-- Tabla de posts
CREATE TABLE posts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL,
  username TEXT NOT NULL,
  caption TEXT,
  image_url TEXT NOT NULL,
  audio_url TEXT,
  likes INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tabla de likes
CREATE TABLE likes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(post_id, user_id)
);

-- Tabla de comentarios
CREATE TABLE comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  username TEXT NOT NULL,
  text TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para optimización
CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_posts_created_at ON posts(created_at DESC);
CREATE INDEX idx_likes_post_id ON likes(post_id);
CREATE INDEX idx_comments_post_id ON comments(post_id);
```

#### Configurar Storage Bucket

1. Ir a **Storage** en Supabase Dashboard
2. Crear bucket: `post-images`
3. Configurar como **público**
4. Política de acceso: permitir uploads autenticados

#### Configurar Post Service

Editar `backend-java/post-service/src/main/resources/application-docker.yml`:

```yaml
supabase:
  url: https://xxxxx.supabase.co  # TU PROJECT URL
  service-role-key: eyJhbGc...     # TU SERVICE ROLE KEY
  storage:
    bucket-name: post-images
```

### 3. Configuración de Red

#### Editar IP del Backend

Todos los servicios deben apuntar a la IP de tu máquina host. Editar:

**Mobile App:**
```dart
// mobile_app/lib/config/api_config.dart
class ApiConfig {
  static const String _baseUrl = 'http://192.168.1.252:8080';  // CAMBIAR IP
}
```

**Docker Compose:**
```yaml
# backend-java/docker-compose.yml
# Ya está configurado para usar red interna (no requiere cambios)
```

---

## 🚀 Despliegue de Servicios Backend

### Opción 1: Despliegue con Docker (Recomendado)

#### Paso 1: Verificar Credenciales

```bash
# Verificar que existe firebase-credentials.json
ls backend-java/auth-service/src/main/resources/firebase-credentials.json

# Verificar configuración de Supabase
cat backend-java/post-service/src/main/resources/application-docker.yml
```

#### Paso 2: Construir Imágenes

```bash
cd backend-java

# Construir todas las imágenes
docker-compose build

# O construir servicios individuales
docker-compose build auth-service
docker-compose build post-service
docker-compose build api-gateway
docker-compose build cuda-backend
```

#### Paso 3: Levantar Servicios

```bash
# Levantar todos los servicios
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f auth-service
```

#### Paso 4: Verificar Estado

```bash
# Ver estado de contenedores
docker ps

# Verificar salud de servicios
curl http://localhost:8080/actuator/health  # API Gateway
curl http://localhost:8082/actuator/health  # Auth Service
curl http://localhost:8081/actuator/health  # Post Service
curl http://localhost:5000/health           # CUDA Backend
```

### Opción 2: Despliegue Manual (Desarrollo)

#### Auth Service

```bash
cd backend-java/auth-service

# Compilar
mvn clean package -DskipTests

# Ejecutar
java -jar target/auth-service-1.0.0.jar --spring.profiles.active=local
```

#### Post Service

```bash
cd backend-java/post-service

# Compilar
mvn clean package -DskipTests

# Ejecutar
java -jar target/post-service-1.0.0.jar --spring.profiles.active=local
```

#### API Gateway

```bash
cd backend-java/api-gateway

# Compilar
mvn clean package -DskipTests

# Ejecutar
java -jar target/api-gateway-1.0.0.jar --spring.profiles.active=local
```

#### CUDA Backend

```bash
cd cuda-service

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# O: .\venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python app/server.py
```

### Comandos Útiles de Docker

```bash
# Detener servicios
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v

# Reconstruir y levantar (tras cambios en código)
docker-compose up -d --build

# Reiniciar un servicio específico
docker-compose restart auth-service

# Ver uso de recursos
docker stats

# Limpiar imágenes antiguas
docker system prune -a
```

---

## 📱 Despliegue de la Aplicación Móvil

### Configuración Inicial

#### 1. Instalar Dependencias

```bash
cd mobile_app

# Obtener dependencias de Flutter
flutter pub get

# Verificar configuración
flutter doctor
```

#### 2. Configurar Firebase

Asegurarse de tener:
- `android/app/google-services.json` (Android)
- `ios/Runner/GoogleService-Info.plist` (iOS)

#### 3. Configurar IP del Backend

Editar `lib/config/api_config.dart`:

```dart
class ApiConfig {
  static const String _baseUrl = 'http://TU_IP:8080';  // Cambiar TU_IP
}
```

### Ejecutar en Modo Desarrollo

```bash
# Ver dispositivos disponibles
flutter devices

# Ejecutar en dispositivo conectado
flutter run

# Ejecutar en modo debug con hot reload
flutter run --debug

# Ejecutar en modo release (mejor rendimiento)
flutter run --release
```

### Construir APK (Android)

```bash
# APK para todas las arquitecturas
flutter build apk

# APK optimizado por arquitectura (más pequeño)
flutter build apk --split-per-abi

# Ubicación del APK
# build/app/outputs/flutter-apk/app-release.apk
```

### Construir para iOS

```bash
# Requiere macOS y Xcode

# Construir IPA
flutter build ios --release

# O abrir en Xcode para firmar y distribuir
open ios/Runner.xcworkspace
```

### Construir para Web

```bash
# Build para web
flutter build web --release

# Servir localmente para probar
cd build/web
python -m http.server 8000
```

### Construir para Windows

```bash
# Build para Windows
flutter build windows --release

# Ejecutable en: build/windows/runner/Release/
```

---

## 🔍 Monitoreo y Salud de Servicios

### Endpoints de Salud

| Servicio | Health Endpoint | Puerto |
|----------|----------------|--------|
| API Gateway | `http://localhost:8080/actuator/health` | 8080 |
| Auth Service | `http://localhost:8082/actuator/health` | 8082 |
| Post Service | `http://localhost:8081/actuator/health` | 8081 |
| CUDA Backend | `http://localhost:5000/health` | 5000 |

### Verificación de Estado

```bash
# Script PowerShell para verificar todos los servicios
function Check-Services {
    $services = @(
        @{Name="API Gateway"; Url="http://localhost:8080/actuator/health"},
        @{Name="Auth Service"; Url="http://localhost:8082/actuator/health"},
        @{Name="Post Service"; Url="http://localhost:8081/actuator/health"},
        @{Name="CUDA Backend"; Url="http://localhost:5000/health"}
    )
    
    foreach ($service in $services) {
        try {
            $response = Invoke-WebRequest -Uri $service.Url -UseBasicParsing
            Write-Host "$($service.Name): ✓ HEALTHY" -ForegroundColor Green
        } catch {
            Write-Host "$($service.Name): ✗ UNHEALTHY" -ForegroundColor Red
        }
    }
}

Check-Services
```

### Logs de Docker

```bash
# Ver logs de todos los servicios
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f auth-service

# Ver últimas 100 líneas
docker-compose logs --tail=100 post-service

# Ver logs con timestamps
docker-compose logs -t api-gateway
```

### Métricas de Actuator

```bash
# Métricas JVM (API Gateway)
curl http://localhost:8080/actuator/metrics

# Métrica específica
curl http://localhost:8080/actuator/metrics/jvm.memory.used

# Info de la aplicación
curl http://localhost:8080/actuator/info
```

---

## 🐛 Solución de Problemas

### Problema 1: Servicios en estado "unhealthy"

**Síntomas:**
```bash
docker ps
# STATUS: Up 2 minutes (unhealthy)
```

**Soluciones:**

1. **Verificar logs del servicio:**
```bash
docker logs upsglam-auth-service
```

2. **Verificar conectividad entre contenedores:**
```bash
docker exec upsglam-api-gateway ping auth-service
```

3. **Reconstruir servicios:**
```bash
docker-compose down
docker-compose build auth-service post-service
docker-compose up -d
```

4. **Verificar credenciales de Firebase:**
```bash
# El archivo debe existir
docker exec upsglam-auth-service ls -la /app/firebase-credentials.json
```

### Problema 2: Error "Connection Timeout" en la App

**Causa**: IP del backend incorrecta o firewall bloqueando

**Soluciones:**

1. **Verificar IP del host:**
```bash
# Windows
ipconfig
# Buscar IPv4 de tu adaptador de red (ej: 192.168.1.252)

# Linux/Mac
ifconfig
# O: ip addr
```

2. **Actualizar IP en la app:**
```dart
// mobile_app/lib/config/api_config.dart
static const String _baseUrl = 'http://192.168.X.X:8080';  // Tu IP real
```

3. **Verificar firewall:**
```bash
# Windows: Permitir puertos 8080, 8081, 8082, 5000
# Ir a: Windows Defender Firewall → Reglas de entrada
```

4. **Probar conectividad desde el móvil:**
```
Abrir navegador en el móvil → http://TU_IP:8080/actuator/health
Debe devolver: {"status":"UP"}
```

### Problema 3: Error al aplicar filtros CUDA

**Síntomas:**
```
Error applying filter: Connection refused
```

**Soluciones:**

1. **Verificar CUDA Backend está corriendo:**
```bash
docker ps | grep cuda-backend
curl http://localhost:5000/health
```

2. **Verificar GPU disponible:**
```bash
docker exec upsglam-cuda-backend nvidia-smi
```

3. **Si no tienes GPU, usar mock service:**
```bash
cd backend-java/pycuda-mock
python app.py  # Mock que simula filtros sin GPU
```

4. **Actualizar URL en gateway a mock:**
```yaml
# backend-java/api-gateway/src/main/resources/application.yml
spring:
  cloud:
    gateway:
      routes:
        - id: cuda-service-apply-filter
          uri: http://localhost:5001  # Mock en puerto 5001
```

### Problema 4: Firebase Authentication Error

**Síntomas:**
```
Error: Invalid credentials
Firebase token verification failed
```

**Soluciones:**

1. **Verificar archivo de credenciales:**
```bash
# Debe ser JSON válido
cat backend-java/auth-service/src/main/resources/firebase-credentials.json
```

2. **Verificar proyecto Firebase activo:**
- Ir a Firebase Console
- Verificar que Authentication esté habilitado
- Verificar que el Service Account Key es del proyecto correcto

3. **Reconstruir con nuevas credenciales:**
```bash
# Copiar nuevo firebase-credentials.json
docker-compose down
docker-compose build auth-service
docker-compose up -d
```

### Problema 5: Posts no se cargan en el Feed

**Causa**: Supabase no configurado o sin datos

**Soluciones:**

1. **Verificar conexión a Supabase:**
```bash
docker logs upsglam-post-service | grep -i supabase
# Debe mostrar: "Successfully connected to Supabase"
```

2. **Verificar configuración:**
```yaml
# backend-java/post-service/src/main/resources/application-docker.yml
supabase:
  url: https://xxxxx.supabase.co  # Debe ser TU URL
  service-role-key: eyJ...         # Debe ser TU KEY
```

3. **Crear datos de prueba:**
- Abrir app móvil
- Crear un post
- Verificar en Supabase Dashboard → Table Editor → posts

4. **Verificar tablas existen:**
```sql
-- Ejecutar en Supabase SQL Editor
SELECT * FROM posts LIMIT 10;
```

### Problema 6: Notificaciones no aparecen

**Causa**: Firestore no configurado correctamente

**Soluciones:**

1. **Verificar Firestore en Firebase Console:**
- Database ID debe ser `(default)` para notificaciones
- Verificar que `db-auth` existe para usuarios

2. **Ver colección de notificaciones:**
- Firebase Console → Firestore Database
- Colección: `notifications`
- Debe tener documentos tras dar like/comment/follow

3. **Verificar logs:**
```bash
docker logs upsglam-post-service | grep -i notification
```

### Problema 7: Censura no funciona

**Causa**: Lista de palabras prohibidas no actualizada

**Solución:**

Editar palabras prohibidas:
```dart
// mobile_app/lib/screens/home/create_post_view.dart
final List<String> _bannedWords = [
  'messi', 
  'barcelona', 
  'visca barca', 
  'barça', 
  'hitler', 
  'nazi',
  'puto',
  'pendejo',
  // Agregar más palabras aquí
];
```

---

## 📡 APIs y Endpoints

### Authentication API (Puerto 8082)

#### Registro
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "username": "johndoe",
  "displayName": "John Doe"
}

Response: 200 OK
{
  "userId": "abc123",
  "email": "user@example.com",
  "username": "johndoe",
  "displayName": "John Doe",
  "token": "eyJhbGc..."
}
```

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}

Response: 200 OK
{
  "userId": "abc123",
  "token": "eyJhbGc..."
}
```

#### Obtener Perfil
```http
GET /api/auth/me
Authorization: Bearer {token}

Response: 200 OK
{
  "userId": "abc123",
  "email": "user@example.com",
  "username": "johndoe",
  "displayName": "John Doe",
  "bio": "",
  "profileImageUrl": ""
}
```

#### Seguir Usuario
```http
POST /api/auth/follow/{userId}
Authorization: Bearer {token}
X-User-Id: {currentUserId}

Response: 200 OK
{
  "success": true,
  "isFollowing": true
}
```

### Posts API (Puerto 8081)

#### Crear Post
```http
POST /api/posts
Authorization: Bearer {token}
X-User-Id: {userId}
X-Username: {username}
Content-Type: application/json

{
  "imageUrl": "https://supabase.co/storage/v1/...",
  "caption": "Mi nuevo post!",
  "audioUrl": "https://supabase.co/storage/v1/..."
}

Response: 201 Created
{
  "id": "uuid",
  "userId": "abc123",
  "username": "johndoe",
  "imageUrl": "...",
  "caption": "Mi nuevo post!",
  "likes": 0,
  "createdAt": "2025-12-12T17:00:00Z"
}
```

#### Obtener Feed
```http
GET /api/feed?page=0&size=20
Authorization: Bearer {token}

Response: 200 OK
{
  "content": [...],
  "page": 0,
  "size": 20,
  "totalElements": 100,
  "totalPages": 5
}
```

#### Dar Like
```http
POST /api/posts/{postId}/like
Authorization: Bearer {token}
X-User-Id: {userId}

Response: 200 OK
{
  "success": true,
  "liked": true,
  "likesCount": 42
}
```

#### Crear Comentario
```http
POST /api/posts/{postId}/comments
Authorization: Bearer {token}
X-User-Id: {userId}
X-Username: {username}
Content-Type: application/json

{
  "text": "Excelente post!"
}

Response: 201 Created
{
  "id": "uuid",
  "postId": "uuid",
  "userId": "abc123",
  "username": "johndoe",
  "text": "Excelente post!",
  "createdAt": "2025-12-12T17:00:00Z"
}
```

#### Obtener Notificaciones
```http
GET /api/notifications/me
Authorization: Bearer {token}
X-User-Id: {userId}

Response: 200 OK
[
  {
    "id": "uuid",
    "type": "like",
    "userId": "abc123",
    "fromUserId": "xyz789",
    "fromUsername": "janedoe",
    "postId": "post-uuid",
    "message": "janedoe le dio like a tu publicación",
    "timestamp": 1702400000000,
    "isRead": false
  }
]
```

### CUDA Filters API (Puerto 5000)

#### Aplicar Filtro
```http
POST /filters/{filterName}
Content-Type: image/jpeg

[Binary Image Data]

Response: 200 OK
Content-Type: image/jpeg

[Processed Image Data]
```

#### Filtros Disponibles
- `ups_logo` - Marca de agua UPS
- `blox_blur` - Desenfoque
- `edge_detection` - Detección de bordes
- `sharpen` - Afilado
- `emboss` - Relieve
- `grayscale` - Escala de grises
- `sepia` - Sepia
- `invert` - Invertir colores

---

## 📊 Estructura de Bases de Datos

### Firebase Firestore

#### Colección: `users` (Database: db-auth)
```json
{
  "userId": "abc123",
  "email": "user@example.com",
  "username": "johndoe",
  "displayName": "John Doe",
  "bio": "Developer",
  "profileImageUrl": "https://...",
  "followers": ["xyz789", "def456"],
  "following": ["xyz789"],
  "createdAt": 1702400000000
}
```

#### Colección: `notifications` (Database: default)
```json
{
  "type": "like|comment|follow",
  "userId": "abc123",        // Recipient
  "fromUserId": "xyz789",    // Sender
  "fromUsername": "janedoe",
  "postId": "uuid",
  "commentText": "...",
  "timestamp": 1702400000000,
  "isRead": false
}
```

### Supabase PostgreSQL

#### Tabla: `posts`
```sql
id          UUID PRIMARY KEY
user_id     TEXT NOT NULL
username    TEXT NOT NULL
caption     TEXT
image_url   TEXT NOT NULL
audio_url   TEXT
likes       INTEGER DEFAULT 0
created_at  TIMESTAMPTZ DEFAULT NOW()
```

#### Tabla: `likes`
```sql
id          UUID PRIMARY KEY
post_id     UUID REFERENCES posts(id)
user_id     TEXT NOT NULL
created_at  TIMESTAMPTZ DEFAULT NOW()
UNIQUE(post_id, user_id)
```

#### Tabla: `comments`
```sql
id          UUID PRIMARY KEY
post_id     UUID REFERENCES posts(id)
user_id     TEXT NOT NULL
username    TEXT NOT NULL
text        TEXT NOT NULL
created_at  TIMESTAMPTZ DEFAULT NOW()
```

---

## 🔐 Seguridad

### Autenticación JWT

Todos los endpoints protegidos requieren:
```http
Authorization: Bearer {firebase-jwt-token}
X-User-Id: {userId}
X-Username: {username}
```

### Censura de Contenido

Palabras prohibidas al crear posts:
- messi, barcelona, visca barca, barça
- hitler, nazi
- puto, pendejo

### Variables de Entorno Sensibles

**Nunca commitear:**
- `firebase-credentials.json`
- `google-services.json`
- Supabase Service Role Keys
- Tokens de acceso

---

## 📈 Escalabilidad y Performance

### Optimizaciones Implementadas

1. **API Gateway**: Reactivo con WebFlux (non-blocking I/O)
2. **Post Service**: R2DBC para acceso reactivo a PostgreSQL
3. **Auth Service**: Firestore con caché de perfiles
4. **Feed**: Paginación (20 posts por página)
5. **Imágenes**: Compresión automática en Supabase Storage
6. **CUDA**: Procesamiento paralelo en GPU

### Recomendaciones para Producción

1. **Usar HTTPS** con certificados SSL/TLS
2. **Configurar CORS** apropiadamente
3. **Rate Limiting** en API Gateway
4. **Monitoring** con Prometheus + Grafana
5. **CDN** para servir imágenes estáticas
6. **Database Connection Pooling**
7. **Redis Cache** para datos frecuentes

---

## 📝 Notas de Versión

### v2.0.0 (Diciembre 2025)

**Nuevas Características:**
- ✅ Sistema de notificaciones backend con Firestore
- ✅ Censura de contenido con palabras prohibidas
- ✅ Arquitectura de microservicios con Docker
- ✅ Procesamiento CUDA de imágenes
- ✅ Feed con scroll infinito
- ✅ Sistema de likes y comentarios reactivo

**Correcciones:**
- 🐛 Fix en autenticación Firebase
- 🐛 Fix en health checks de servicios
- 🐛 Fix en notificaciones cross-device

---

## 📞 Soporte y Contacto

**Equipo de Desarrollo:**
- Universidad Politécnica Salesiana
- Proyecto: UPSGlam 2.0

**Reportar Issues:**
- Crear issue en repositorio GitHub
- Incluir logs relevantes
- Describir pasos para reproducir

---

## 📜 Licencia

Este proyecto es desarrollado con fines educativos para la Universidad Politécnica Salesiana.

---

**Última actualización:** Diciembre 12, 2025
