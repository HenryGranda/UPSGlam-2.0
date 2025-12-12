# 🔐 Auth Service - UPSGlam 2.0

## 📋 Descripción General

El **Auth Service** es el microservicio responsable de la autenticación, autorización y gestión de usuarios en la plataforma UPSGlam. Implementado con **Spring Boot WebFlux** (reactivo) y **Firebase Authentication + Firestore**, proporciona endpoints para registro, login, gestión de perfiles de usuario y sistema de seguimientos (follows).

---

## 🏗️ Arquitectura

### Stack Tecnológico
- **Framework**: Spring Boot 3.2.0 (Reactive WebFlux)
- **Runtime**: Java 21 (LTS)
- **Authentication**: Firebase Admin SDK 9.2.0
- **Database**: Google Cloud Firestore
- **Storage**: Firebase Cloud Storage
- **Build Tool**: Maven 3.9+
- **Container**: Docker (eclipse-temurin:21-jre)

### Dependencias Principales
```xml
<!-- Spring Boot WebFlux (Reactive) -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-webflux</artifactId>
</dependency>

<!-- Firebase Admin SDK -->
<dependency>
    <groupId>com.google.firebase</groupId>
    <artifactId>firebase-admin</artifactId>
    <version>9.2.0</version>
</dependency>

<!-- Google Cloud Firestore -->
<dependency>
    <groupId>com.google.cloud</groupId>
    <artifactId>google-cloud-firestore</artifactId>
</dependency>

<!-- Google Cloud Storage -->
<dependency>
    <groupId>com.google.cloud</groupId>
    <artifactId>google-cloud-storage</artifactId>
</dependency>
```

---

## 🚀 Características Principales

### 1. **Autenticación con Firebase**
- Registro de usuarios con email/password
- Login y gestión de sesiones
- Verificación de tokens JWT
- Gestión de refresh tokens

### 2. **Gestión de Usuarios**
- CRUD completo de perfiles de usuario
- Upload de avatares a Firebase Storage
- Actualización de información de perfil
- Búsqueda de usuarios

### 3. **Sistema de Seguimientos (Follows)**
- Follow/Unfollow de usuarios
- Obtener lista de followers
- Obtener lista de following
- Verificar estado de seguimiento

### 4. **Firestore Database**
- Colección `users`: Datos de perfil
- Colección `follows`: Relaciones de seguimiento
- Queries optimizadas con índices
- Operaciones reactivas (non-blocking)

---

## 🗂️ Estructura del Proyecto

```
auth-service/
├── src/
│   └── main/
│       ├── java/ec/ups/upsglam/auth/
│       │   ├── api/
│       │   │   ├── controller/      # REST Controllers
│       │   │   ├── dto/             # Data Transfer Objects
│       │   │   └── handler/         # Exception Handlers
│       │   ├── config/              # Configuraciones
│       │   │   └── FirebaseConfig.java
│       │   ├── domain/
│       │   │   ├── model/           # Domain Models
│       │   │   ├── repository/      # Repository Interfaces
│       │   │   └── service/         # Business Logic
│       │   ├── infrastructure/
│       │   │   └── firebase/        # Firebase Implementation
│       │   └── AuthServiceApplication.java
│       └── resources/
│           ├── application.yml
│           ├── application-docker.yml
│           └── application-local.yml
├── docs/                            # Documentación adicional
├── Dockerfile
├── pom.xml
└── README.md
```

---

## 🔧 Configuración

### Variables de Entorno

| Variable | Descripción | Ejemplo | Requerido |
|----------|-------------|---------|-----------|
| `SERVER_PORT` | Puerto del servicio | `8082` | ❌ |
| `SPRING_PROFILES_ACTIVE` | Perfil activo | `docker` | ✅ |
| `FIREBASE_PROJECT_ID` | ID del proyecto Firebase | `upsglam-8c88f` | ✅ |
| `FIREBASE_API_KEY` | API Key de Firebase | `AIza...` | ✅ |
| `FIREBASE_CREDENTIALS_PATH` | Ruta a credenciales JSON | `/app/firebase-credentials.json` | ✅ |
| `FIREBASE_DATABASE_ID` | ID de la base de datos Firestore | `db-auth` | ❌ |
| `FIREBASE_STORAGE_BUCKET` | Bucket de storage | `upsglam-8c88f.appspot.com` | ❌ |
| `JAVA_OPTS` | Opciones JVM | `-Xmx512m -Xms256m` | ❌ |

### Firebase Credentials

#### Obtener `firebase-credentials.json`:
1. Ir a [Firebase Console](https://console.firebase.google.com/)
2. Seleccionar tu proyecto
3. Settings → Service Accounts
4. Click "Generate new private key"
5. Guardar el archivo como `firebase-credentials.json`

⚠️ **IMPORTANTE**: Este archivo contiene credenciales sensibles. NUNCA lo subas a git.

---

## 📡 API Endpoints

### Base URL
- **Local**: `http://localhost:8082/api`
- **Docker**: `http://auth-service:8082/api`
- **Gateway**: `http://localhost:8080/api/auth`

### 1. Health & Info

#### `GET /health`
Health check del servicio.

**Response:**
```json
{
  "status": "UP",
  "timestamp": "2025-12-12T10:30:00Z"
}
```

---

### 2. Authentication

#### `POST /auth/register`
Registrar nuevo usuario.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepass123",
  "username": "johndoe",
  "displayName": "John Doe"
}
```

**Response:**
```json
{
  "userId": "firebase-uid-123",
  "email": "user@example.com",
  "username": "johndoe",
  "displayName": "John Doe",
  "photoUrl": null,
  "createdAt": "2025-12-12T10:30:00Z"
}
```

#### `POST /auth/login`
Iniciar sesión.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepass123"
}
```

**Response:**
```json
{
  "idToken": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refreshToken": "AOk...",
  "expiresIn": "3600",
  "userId": "firebase-uid-123"
}
```

#### `GET /auth/me`
Obtener usuario actual (requiere token).

**Headers:**
```
Authorization: Bearer <idToken>
```

**Response:**
```json
{
  "userId": "firebase-uid-123",
  "email": "user@example.com",
  "username": "johndoe",
  "displayName": "John Doe",
  "photoUrl": "https://storage.url/avatar.jpg",
  "bio": "Developer & Photographer",
  "followersCount": 150,
  "followingCount": 200
}
```

---

### 3. User Management

#### `GET /users/{userId}`
Obtener perfil de usuario por ID.

**Response:**
```json
{
  "userId": "firebase-uid-123",
  "username": "johndoe",
  "displayName": "John Doe",
  "photoUrl": "https://storage.url/avatar.jpg",
  "bio": "Developer & Photographer",
  "followersCount": 150,
  "followingCount": 200,
  "postsCount": 45
}
```

#### `PUT /users/{userId}`
Actualizar perfil de usuario.

**Request:**
```json
{
  "displayName": "John Updated",
  "bio": "New bio text",
  "photoUrl": "https://new-url.com/avatar.jpg"
}
```

#### `POST /users/{userId}/avatar`
Upload de avatar (multipart/form-data).

**Request:**
```bash
curl -X POST http://localhost:8082/api/users/user123/avatar \
  -F "avatar=@avatar.jpg"
```

**Response:**
```json
{
  "photoUrl": "https://storage.url/avatars/user123-1234567890.jpg"
}
```

---

### 4. Follow System

#### `POST /users/{userId}/follow`
Seguir a un usuario.

**Headers:**
```
X-User-Id: current-user-id
```

**Response:**
```json
{
  "success": true,
  "message": "Now following user123"
}
```

#### `DELETE /users/{userId}/unfollow`
Dejar de seguir a un usuario.

**Response:**
```json
{
  "success": true,
  "message": "Unfollowed user123"
}
```

#### `GET /users/{userId}/followers`
Obtener lista de followers.

**Response:**
```json
{
  "followers": [
    {
      "userId": "user1",
      "username": "alice",
      "displayName": "Alice",
      "photoUrl": "https://..."
    }
  ],
  "total": 150
}
```

#### `GET /users/{userId}/following`
Obtener lista de usuarios seguidos.

#### `GET /users/{userId}/is-following/{targetUserId}`
Verificar si sigue a un usuario.

**Response:**
```json
{
  "isFollowing": true
}
```

---

## 🗄️ Firestore Schema

### Collection: `users`

```javascript
{
  "userId": "firebase-uid-123",        // Document ID
  "email": "user@example.com",
  "username": "johndoe",               // Unique
  "displayName": "John Doe",
  "photoUrl": "https://storage.url/avatar.jpg",
  "bio": "Developer & Photographer",
  "followersCount": 150,
  "followingCount": 200,
  "postsCount": 45,
  "createdAt": "2025-12-12T10:30:00Z",
  "updatedAt": "2025-12-12T10:30:00Z"
}
```

**Indexes:**
- `username` (unique)
- `email` (unique)
- `createdAt` (descending)

### Collection: `follows`

```javascript
{
  "followId": "user1_user2",           // Document ID: follower_following
  "followerId": "user1",               // User who follows
  "followingId": "user2",              // User being followed
  "createdAt": "2025-12-12T10:30:00Z"
}
```

**Indexes:**
- `followerId` + `createdAt` (descending)
- `followingId` + `createdAt` (descending)

---

## 🛠️ Desarrollo Local

### Prerrequisitos
- Java 21 JDK
- Maven 3.9+
- Firebase Project configurado
- `firebase-credentials.json` en el directorio raíz

### Setup

```bash
# 1. Clonar repositorio
cd backend-java/auth-service

# 2. Copiar credenciales
cp /path/to/firebase-credentials.json ./firebase-credentials.json

# 3. Configurar application-local.yml
cp src/main/resources/application-local.yml.example \
   src/main/resources/application-local.yml

# 4. Editar application-local.yml con tus valores
```

### Compilación

```bash
# Limpiar y compilar
mvn clean package

# Compilar sin tests
mvn clean package -DskipTests

# Tests
mvn test
```

### Ejecución Local

```bash
# Método 1: Maven
mvn spring-boot:run -Dspring-boot.run.profiles=local

# Método 2: JAR
java -jar target/auth-service-1.0.0.jar --spring.profiles.active=local

# Método 3: PowerShell script
.\start-auth.ps1
```

---

## 🐳 Docker

### Dockerfile

```dockerfile
# Multi-stage build
FROM maven:3.9-eclipse-temurin-21 AS builder
WORKDIR /app
COPY pom.xml .
COPY src ./src
RUN mvn -q clean package -DskipTests

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=builder /app/target/auth-service-*.jar app.jar

ENV SERVER_PORT=8082
ENV SPRING_PROFILES_ACTIVE=docker
ENV JAVA_OPTS="-Xmx512m -Xms256m"

EXPOSE 8082

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar --spring.profiles.active=${SPRING_PROFILES_ACTIVE}"]
```

### Build & Run

```bash
# Build image
docker build -t upsglam-auth-service:latest .

# Run container
docker run -d \
  --name auth-service \
  -p 8082:8082 \
  -e SPRING_PROFILES_ACTIVE=docker \
  -e FIREBASE_PROJECT_ID=your-project-id \
  -e FIREBASE_API_KEY=your-api-key \
  -v $(pwd)/firebase-credentials.json:/app/firebase-credentials.json:ro \
  --network upsglam-network \
  upsglam-auth-service:latest

# Logs
docker logs -f auth-service

# Stop
docker stop auth-service && docker rm auth-service
```

### Docker Compose

```yaml
auth-service:
  build:
    context: ./auth-service
    dockerfile: Dockerfile
  container_name: upsglam-auth-service
  ports:
    - "8082:8082"
  environment:
    - SERVER_PORT=8082
    - SPRING_PROFILES_ACTIVE=docker
    - FIREBASE_PROJECT_ID=${FIREBASE_PROJECT_ID}
    - FIREBASE_API_KEY=${FIREBASE_API_KEY}
    - JAVA_OPTS=-Xmx512m -Xms256m
  volumes:
    - ./firebase-credentials.json:/app/firebase-credentials.json:ro
  networks:
    - upsglam-network
```

---

## 🧪 Testing

### Scripts de Prueba

```powershell
# Test básico
.\test-auth.ps1

# Test de follows
.\test-follows.ps1

# Test completo de follows
.\test-follows-complete.ps1
```

### Ejemplos de Testing

```bash
# 1. Health check
curl http://localhost:8082/api/health

# 2. Registro
curl -X POST http://localhost:8082/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123",
    "username": "testuser"
  }'

# 3. Login
curl -X POST http://localhost:8082/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123"
  }'

# 4. Get user
curl http://localhost:8082/api/users/user123

# 5. Follow user
curl -X POST http://localhost:8082/api/users/user456/follow \
  -H "X-User-Id: user123"
```

---

## 🔒 Seguridad

### Firebase Authentication
- Passwords hasheados automáticamente por Firebase
- Tokens JWT con expiración de 1 hora
- Refresh tokens para renovación

### Best Practices Implementadas
- ✅ Credenciales en variables de entorno
- ✅ `firebase-credentials.json` en .gitignore
- ✅ Validación de inputs con Bean Validation
- ✅ Headers de seguridad (CORS, CSP)
- ✅ Logs sanitizados (sin passwords)

### CORS Configuration
```yaml
spring:
  webflux:
    cors:
      allowed-origins: "*"
      allowed-methods: "*"
      allowed-headers: "*"
```

---

## 📊 Rendimiento

### Configuración de Memoria
```bash
# Recomendado para producción
JAVA_OPTS="-Xmx512m -Xms256m -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

### Métricas Esperadas
- **Latencia promedio**: < 100ms
- **P95 latency**: < 250ms
- **Throughput**: > 1000 req/s
- **Memory usage**: ~300-400MB

---

## 🐛 Troubleshooting

### Problema: Firebase credentials not found

```bash
# Verificar que el archivo existe
ls -la firebase-credentials.json

# Verificar permisos
chmod 600 firebase-credentials.json

# Verificar path en config
grep -r "firebase.credentials.path" src/main/resources/
```

### Problema: Connection timeout to Firestore

```yaml
# Aumentar timeouts
firebase:
  timeout:
    connect: 10000
    read: 30000
```

### Problema: Out of Memory

```bash
# Aumentar heap size
JAVA_OPTS="-Xmx1024m -Xms512m"
```

---

## 📚 Referencias

- [Firebase Admin SDK Java](https://firebase.google.com/docs/admin/setup)
- [Spring Boot WebFlux](https://docs.spring.io/spring-framework/reference/web/webflux.html)
- [Cloud Firestore](https://firebase.google.com/docs/firestore)
- [Project Main README](../../README.md)

---

## 📝 Changelog

### Version 1.0.0
- ✅ Firebase Authentication integration
- ✅ User CRUD operations
- ✅ Follow/Unfollow system
- ✅ Avatar upload to Firebase Storage
- ✅ Firestore database integration
- ✅ Docker support
- ✅ Health checks y monitoring

---

## 👥 Autor

UPSGlam Development Team - Universidad Politécnica Salesiana
