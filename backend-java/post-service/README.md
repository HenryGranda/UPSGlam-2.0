# 📸 Post Service - UPSGlam 2.0

## 📋 Descripción General

El **Post Service** es el microservicio central de UPSGlam responsable de la gestión completa de publicaciones, incluyendo posts, imágenes, likes y comentarios. Implementado con **Spring Boot WebFlux** (reactivo), utiliza **Firestore** para datos NoSQL, **Supabase Storage** para almacenamiento de imágenes, **Supabase Postgres (R2DBC)** para datos relacionales, y se integra con el **CUDA Backend** para procesamiento de imágenes.

---

## 🏗️ Arquitectura

### Stack Tecnológico
- **Framework**: Spring Boot 3.2.0 (Reactive WebFlux)
- **Runtime**: Java 21 (LTS)
- **NoSQL Database**: Google Cloud Firestore
- **SQL Database**: Supabase PostgreSQL + R2DBC (Reactive)
- **Object Storage**: Supabase Storage
- **Authentication**: Firebase Admin SDK 9.2.0
- **Image Processing**: PyCUDA Service (HTTP Client)
- **Build Tool**: Maven 3.9+
- **Container**: Docker (eclipse-temurin:21-jre)

### Dependencias Principales
```xml
<!-- Spring Boot WebFlux (Reactive) -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-webflux</artifactId>
</dependency>

<!-- R2DBC PostgreSQL (Reactive Database) -->
<dependency>
    <groupId>io.r2dbc</groupId>
    <artifactId>r2dbc-postgresql</artifactId>
</dependency>

<!-- Firebase Admin SDK -->
<dependency>
    <groupId>com.google.firebase</groupId>
    <artifactId>firebase-admin</artifactId>
    <version>9.2.0</version>
</dependency>

<!-- Supabase Client -->
<dependency>
    <groupId>io.github.jan-tennert.supabase</groupId>
    <artifactId>supabase-kt</artifactId>
    <version>1.1.3</version>
</dependency>
```

---

## 🚀 Características Principales

### 1. **Gestión de Posts**
- CRUD completo de publicaciones
- Upload de imágenes a Supabase Storage
- Aplicación de filtros CUDA
- Obtención de feed paginado
- Posts por usuario
- Eliminación de posts con cascade

### 2. **Sistema de Likes**
- Dar/quitar like a posts
- Contador de likes en tiempo real
- Verificar si usuario dio like
- Lista de usuarios que dieron like

### 3. **Sistema de Comentarios**
- Agregar comentarios a posts
- Editar/eliminar comentarios
- Comentarios anidados (replies)
- Paginación de comentarios

### 4. **Procesamiento de Imágenes**
- Integración con CUDA Backend
- Aplicación de filtros GPU
- Preview de filtros
- Upload temporal y final

### 5. **Base de Datos Híbrida**

#### Firestore (NoSQL)
- **Posts**: Metadata, caption, filter
- **Likes**: Relaciones user-post
- **Comments**: Comentarios y replies

#### Supabase PostgreSQL (Relacional)
- **Analytics**: Métricas y estadísticas
- **Reports**: Reportes de contenido
- **Relationships**: Datos relacionales complejos

---

## 🗂️ Estructura del Proyecto

```
post-service/
├── src/
│   └── main/
│       ├── java/ec/ups/upsglam/post/
│       │   ├── api/
│       │   │   ├── controller/      # REST Controllers
│       │   │   │   ├── PostController.java
│       │   │   │   ├── LikeController.java
│       │   │   │   └── CommentController.java
│       │   │   ├── dto/             # Data Transfer Objects
│       │   │   └── handler/         # Exception Handlers
│       │   ├── config/              # Configuraciones
│       │   │   ├── FirebaseConfig.java
│       │   │   ├── SupabaseConfig.java
│       │   │   └── WebClientConfig.java
│       │   ├── domain/
│       │   │   ├── model/           # Domain Models
│       │   │   ├── repository/      # Repository Interfaces
│       │   │   └── service/         # Business Logic
│       │   ├── infrastructure/
│       │   │   ├── firestore/       # Firestore Implementation
│       │   │   ├── supabase/        # Supabase Implementation
│       │   │   └── pycuda/          # PyCUDA Client
│       │   └── PostServiceApplication.java
│       └── resources/
│           ├── application.yml
│           ├── application-docker.yml
│           └── application-local.yml
├── docs/                            # Documentación adicional
│   ├── API-TESTS.md
│   ├── SUPABASE-SETUP.md
│   ├── SECURITY.md
│   └── TESTING-GUIDE.md
├── Dockerfile
├── pom.xml
└── README.md
```

---

## 🔧 Configuración

### Variables de Entorno

| Variable | Descripción | Ejemplo | Requerido |
|----------|-------------|---------|-----------|
| `SERVER_PORT` | Puerto del servicio | `8081` | ❌ |
| `SPRING_PROFILES_ACTIVE` | Perfil activo | `docker` | ✅ |
| `FIREBASE_PROJECT_ID` | ID proyecto Firebase | `upsglam-8c88f` | ✅ |
| `FIREBASE_API_KEY` | API Key Firebase | `AIza...` | ✅ |
| `FIREBASE_DATABASE_ID` | ID database Firestore | `db-auth` | ✅ |
| `SUPABASE_URL` | URL de Supabase | `https://xxx.supabase.co` | ✅ |
| `SUPABASE_KEY` | Anon key de Supabase | `eyJhbGc...` | ✅ |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key | `eyJhbGc...` | ✅ |
| `SUPABASE_DB_HOST` | Host Postgres | `xxx.pooler.supabase.com` | ✅ |
| `SUPABASE_DB_PORT` | Puerto Postgres | `6543` | ❌ |
| `SUPABASE_DB_NAME` | Nombre BD | `postgres` | ❌ |
| `SUPABASE_DB_USER` | Usuario BD | `postgres.xxx` | ✅ |
| `SUPABASE_DB_PASSWORD` | Password BD | `password` | ✅ |
| `PYCUDA_SERVICE_URL` | URL CUDA Backend | `http://cuda-backend:5000` | ✅ |
| `JAVA_OPTS` | Opciones JVM | `-Xmx512m -Xms256m` | ❌ |

### Setup Supabase

Ver [SUPABASE-SETUP.md](./docs/SUPABASE-SETUP.md) para instrucciones detalladas.

#### 1. Crear Proyecto Supabase
1. Ir a [supabase.com](https://supabase.com)
2. Crear nuevo proyecto
3. Copiar URL, anon key y service role key

#### 2. Configurar Storage
```sql
-- Crear bucket público
INSERT INTO storage.buckets (id, name, public)
VALUES ('upsglam', 'upsglam', true);

-- Políticas de acceso
CREATE POLICY "Public read access"
ON storage.objects FOR SELECT
USING (bucket_id = 'upsglam');

CREATE POLICY "Authenticated users can upload"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'upsglam' AND auth.role() = 'authenticated');
```

#### 3. Configurar Database (Opcional)
```sql
-- Tabla de analytics (ejemplo)
CREATE TABLE post_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id TEXT NOT NULL,
    views INT DEFAULT 0,
    shares INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_post_analytics_post_id ON post_analytics(post_id);
```

---

## 📡 API Endpoints

### Base URL
- **Local**: `http://localhost:8081/api`
- **Docker**: `http://post-service:8081/api`
- **Gateway**: `http://localhost:8080/api`

---

### 1. Health & Info

#### `GET /health`
Health check del servicio.

**Response:**
```json
{
  "status": "UP",
  "firestore": "connected",
  "supabase": "connected",
  "pycuda": "available"
}
```

---

### 2. Image Management

#### `POST /images/upload`
Upload de imagen a Supabase Storage.

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:8081/api/images/upload \
  -H "X-User-Id: user123" \
  -F "image=@photo.jpg"
```

**Response:**
```json
{
  "imageId": "user123-1733614800000.jpg",
  "imageUrl": "https://xxx.supabase.co/storage/v1/object/public/upsglam/posts/user123-1733614800000.jpg",
  "uploadedAt": "2025-12-12T10:30:00Z"
}
```

#### `POST /images/preview`
Preview de imagen con filtro CUDA (sin guardar).

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:8081/api/images/preview \
  -F "image=@photo.jpg" \
  -F "filter=gaussian"
```

**Response:**
- Content-Type: `image/jpeg`
- Body: Imagen con filtro aplicado

---

### 3. Posts CRUD

#### `POST /posts`
Crear nuevo post.

**Headers:**
```
X-User-Id: user123
X-Username: johndoe
Content-Type: application/json
```

**Request:**
```json
{
  "imageUrl": "https://xxx.supabase.co/storage/.../image.jpg",
  "caption": "¡Hermoso atardecer en Quito! 🌅",
  "filter": "gaussian"
}
```

**Response:**
```json
{
  "postId": "post_abc123",
  "userId": "user123",
  "username": "johndoe",
  "imageUrl": "https://...",
  "caption": "¡Hermoso atardecer en Quito! 🌅",
  "filter": "gaussian",
  "likesCount": 0,
  "commentsCount": 0,
  "createdAt": "2025-12-12T10:30:00Z"
}
```

#### `GET /feed?page=0&size=20`
Obtener feed de posts paginado.

**Query Parameters:**
- `page`: Número de página (default: 0)
- `size`: Tamaño de página (default: 20)

**Response:**
```json
{
  "posts": [
    {
      "postId": "post_abc123",
      "userId": "user123",
      "username": "johndoe",
      "userPhotoUrl": "https://...",
      "imageUrl": "https://...",
      "caption": "¡Hermoso atardecer! 🌅",
      "filter": "gaussian",
      "likesCount": 42,
      "commentsCount": 5,
      "createdAt": "2025-12-12T10:30:00Z"
    }
  ],
  "page": 0,
  "size": 20,
  "totalPages": 5,
  "totalElements": 100
}
```

#### `GET /posts/{postId}`
Obtener post por ID.

**Response:**
```json
{
  "postId": "post_abc123",
  "userId": "user123",
  "username": "johndoe",
  "userPhotoUrl": "https://...",
  "imageUrl": "https://...",
  "caption": "¡Hermoso atardecer! 🌅",
  "filter": "gaussian",
  "likesCount": 42,
  "commentsCount": 5,
  "createdAt": "2025-12-12T10:30:00Z",
  "updatedAt": "2025-12-12T10:30:00Z"
}
```

#### `DELETE /posts/{postId}`
Eliminar post.

**Headers:**
```
X-User-Id: user123
```

**Response:**
```json
{
  "success": true,
  "message": "Post deleted successfully"
}
```

#### `PATCH /posts/{postId}/caption`
Actualizar caption del post.

**Request:**
```json
{
  "caption": "Nueva descripción actualizada 📸"
}
```

#### `GET /posts/user/{userId}?page=0&size=20`
Obtener posts de un usuario específico.

---

### 4. Likes

#### `POST /posts/{postId}/likes`
Dar like a un post.

**Headers:**
```
X-User-Id: user123
X-Username: johndoe
```

**Response:**
```json
{
  "success": true,
  "likesCount": 43,
  "message": "Like added"
}
```

#### `DELETE /posts/{postId}/likes`
Quitar like de un post.

**Response:**
```json
{
  "success": true,
  "likesCount": 42,
  "message": "Like removed"
}
```

#### `GET /posts/{postId}/likes`
Obtener lista de usuarios que dieron like.

**Response:**
```json
{
  "likes": [
    {
      "userId": "user456",
      "username": "alice",
      "photoUrl": "https://...",
      "likedAt": "2025-12-12T11:00:00Z"
    }
  ],
  "total": 42
}
```

#### `GET /posts/{postId}/likes/check?userId=user123`
Verificar si un usuario dio like.

**Response:**
```json
{
  "hasLiked": true
}
```

---

### 5. Comments

#### `POST /posts/{postId}/comments`
Agregar comentario a un post.

**Headers:**
```
X-User-Id: user123
X-Username: johndoe
```

**Request:**
```json
{
  "text": "¡Increíble foto! 📸",
  "parentCommentId": null
}
```

**Response:**
```json
{
  "commentId": "comment_xyz789",
  "postId": "post_abc123",
  "userId": "user123",
  "username": "johndoe",
  "userPhotoUrl": "https://...",
  "text": "¡Increíble foto! 📸",
  "createdAt": "2025-12-12T11:30:00Z"
}
```

#### `GET /posts/{postId}/comments?page=0&size=20`
Obtener comentarios de un post.

**Response:**
```json
{
  "comments": [
    {
      "commentId": "comment_xyz789",
      "userId": "user456",
      "username": "alice",
      "userPhotoUrl": "https://...",
      "text": "¡Increíble foto! 📸",
      "repliesCount": 2,
      "createdAt": "2025-12-12T11:30:00Z"
    }
  ],
  "total": 5
}
```

#### `PATCH /comments/{commentId}`
Actualizar comentario.

**Request:**
```json
{
  "text": "Texto actualizado del comentario"
}
```

#### `DELETE /comments/{commentId}`
Eliminar comentario.

---

## 🗄️ Firestore Schema

### Collection: `posts`

```javascript
{
  "postId": "post_abc123",           // Document ID
  "userId": "firebase-uid-123",
  "username": "johndoe",
  "userPhotoUrl": "https://...",
  "imageUrl": "https://supabase.co/.../image.jpg",
  "caption": "¡Hermoso atardecer! 🌅",
  "filter": "gaussian",              // Filter name
  "likesCount": 42,
  "commentsCount": 5,
  "createdAt": "2025-12-12T10:30:00Z",
  "updatedAt": "2025-12-12T10:30:00Z"
}
```

**Indexes:**
- `userId` + `createdAt` (descending)
- `createdAt` (descending) - Para feed global

### Collection: `likes`

```javascript
{
  "likeId": "post_abc123_user123",   // Document ID: postId_userId
  "postId": "post_abc123",
  "userId": "user123",
  "username": "johndoe",
  "createdAt": "2025-12-12T11:00:00Z"
}
```

**Indexes:**
- `postId` + `createdAt` (descending)
- `userId` + `createdAt` (descending)

### Collection: `comments`

```javascript
{
  "commentId": "comment_xyz789",     // Document ID
  "postId": "post_abc123",
  "userId": "user456",
  "username": "alice",
  "userPhotoUrl": "https://...",
  "text": "¡Increíble foto! 📸",
  "parentCommentId": null,           // null = top-level comment
  "repliesCount": 2,
  "createdAt": "2025-12-12T11:30:00Z",
  "updatedAt": "2025-12-12T11:30:00Z"
}
```

**Indexes:**
- `postId` + `createdAt` (ascending)
- `parentCommentId` + `createdAt` (ascending)

---

## 🛠️ Desarrollo Local

### Prerrequisitos
- Java 21 JDK
- Maven 3.9+
- Firebase Project configurado
- Supabase Project configurado
- CUDA Backend corriendo (port 5000)

### Setup

```bash
# 1. Clonar repositorio
cd backend-java/post-service

# 2. Copiar credenciales Firebase
cp /path/to/firebase-credentials.json ./firebase-credentials.json

# 3. Copiar y configurar application-local.yml
cp src/main/resources/application-local.yml.example \
   src/main/resources/application-local.yml

# 4. Editar con tus credenciales de Supabase
nano src/main/resources/application-local.yml
```

### Compilación

```bash
# Limpiar y compilar
mvn clean package

# Sin tests
mvn clean package -DskipTests

# Tests
mvn test
```

### Ejecución Local

```bash
# Maven
mvn spring-boot:run -Dspring-boot.run.profiles=local

# JAR
java -jar target/post-service-1.0.0.jar --spring.profiles.active=local

# PowerShell
.\start-post-service.ps1
```

---

## 🐳 Docker

### Dockerfile

```dockerfile
FROM maven:3.9-eclipse-temurin-21 AS builder
WORKDIR /app
COPY pom.xml .
COPY src ./src
RUN mvn -q clean package -DskipTests

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=builder /app/target/post-service-*.jar app.jar

ENV SERVER_PORT=8081
EXPOSE 8081

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar --spring.profiles.active=${SPRING_PROFILES_ACTIVE}"]
```

### Docker Commands

```bash
# Build
docker build -t upsglam-post-service:latest .

# Run
docker run -d \
  --name post-service \
  -p 8081:8081 \
  -e SPRING_PROFILES_ACTIVE=docker \
  -e FIREBASE_PROJECT_ID=your-project \
  -e SUPABASE_URL=https://xxx.supabase.co \
  -v $(pwd)/firebase-credentials.json:/app/firebase-credentials.json:ro \
  --network upsglam-network \
  upsglam-post-service:latest
```

---

## 🧪 Testing

### Scripts de Prueba

```powershell
# Test básico
.\test-simple.ps1

# Test de endpoints
.\test-endpoints.ps1

# Test de Firestore
.\test-firestore.ps1

# Test de API completo
.\test-api.ps1
```

Ver [TESTING-GUIDE.md](./docs/TESTING-GUIDE.md) para más detalles.

---

## 📊 Rendimiento

### Configuración JVM
```bash
JAVA_OPTS="-Xmx512m -Xms256m -XX:+UseG1GC"
```

### Métricas
- **Latencia promedio**: < 150ms
- **Upload imagen**: < 2s
- **Apply filter**: < 5s (depende de CUDA)

---

## 🔒 Seguridad

Ver [SECURITY.md](./docs/SECURITY.md) para detalles completos.

---

## 📚 Referencias

- [Firebase Firestore](https://firebase.google.com/docs/firestore)
- [Supabase Storage](https://supabase.com/docs/guides/storage)
- [R2DBC Documentation](https://r2dbc.io/)
- [API Tests](./docs/API-TESTS.md)

---

## 👥 Autor

UPSGlam Development Team - Universidad Politécnica Salesiana
