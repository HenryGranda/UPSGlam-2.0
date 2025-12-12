# 🌐 API Gateway - UPSGlam 2.0

## 📋 Descripción General

El API Gateway es el punto de entrada único para todas las peticiones del sistema UPSGlam. Implementado con **Spring Cloud Gateway** (reactivo), este servicio se encarga del enrutamiento, balanceo de carga, y gestión centralizada de peticiones hacia los microservicios backend.

---

## 🏗️ Arquitectura

### Stack Tecnológico
- **Framework**: Spring Boot 3.2.0
- **Gateway**: Spring Cloud Gateway 2023.0.0
- **Runtime**: Java 21 (LTS)
- **Paradigma**: Programación Reactiva (WebFlux)
- **Build Tool**: Maven 3.9+
- **Container**: Docker (eclipse-temurin:21-jre)

### Dependencias Principales
```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-gateway</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-webflux</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

---

## 🚀 Características Principales

### 1. **Enrutamiento Dinámico**
- Enrutamiento basado en paths hacia microservicios específicos
- Reescritura de URLs automática
- Configuración declarativa en `application.yml`

### 2. **Balanceo de Carga**
- Load balancing automático cuando hay múltiples instancias
- Health checks de servicios downstream

### 3. **Gestión de Headers**
- Propagación de headers de autenticación
- Inyección de headers personalizados
- CORS configuration

### 4. **Monitoreo**
- Spring Boot Actuator para health checks
- Endpoints `/health`, `/info`, `/metrics`
- Logs estructurados

---

## 🔧 Configuración

### Variables de Entorno

| Variable | Descripción | Default | Requerido |
|----------|-------------|---------|-----------|
| `SERVER_PORT` | Puerto del gateway | `8080` | ❌ |
| `SPRING_PROFILES_ACTIVE` | Perfil activo | `default` | ❌ |
| `JAVA_OPTS` | Opciones JVM | `-Xmx256m -Xms128m` | ❌ |

### Archivo de Configuración: `application.yml`

```yaml
server:
  port: 8080

spring:
  application:
    name: api-gateway
  cloud:
    gateway:
      routes:
        # Auth Service Routes
        - id: auth-service
          uri: http://auth-service:8082
          predicates:
            - Path=/api/auth/**, /api/users/**
          filters:
            - RewritePath=/api/(?<segment>.*), /$\{segment}
        
        # Post Service Routes
        - id: post-service
          uri: http://post-service:8081
          predicates:
            - Path=/api/posts/**, /api/feed/**, /api/images/**
          filters:
            - RewritePath=/api/(?<segment>.*), /$\{segment}
        
        # CUDA Service Routes
        - id: cuda-service
          uri: http://cuda-backend:5000
          predicates:
            - Path=/api/filters/**
          filters:
            - RewritePath=/api/filters/(?<segment>.*), /$\{segment}
```

---

## 📡 Rutas y Endpoints

### Arquitectura de Enrutamiento

```
Client Request → API Gateway (8080) → Backend Services
                        ↓
        ┌───────────────┼───────────────┬──────────────┐
        │               │               │              │
   Auth Service    Post Service   CUDA Service   Other Services
   (Port 8082)     (Port 8081)    (Port 5000)
```

### Tabla de Rutas

| Path Pattern | Target Service | Target Port | Description |
|--------------|----------------|-------------|-------------|
| `/api/auth/**` | auth-service | 8082 | Autenticación y usuarios |
| `/api/users/**` | auth-service | 8082 | Gestión de perfiles |
| `/api/posts/**` | post-service | 8081 | CRUD de posts |
| `/api/feed/**` | post-service | 8081 | Feed de publicaciones |
| `/api/images/**` | post-service | 8081 | Upload y gestión de imágenes |
| `/api/filters/**` | cuda-backend | 5000 | Filtros CUDA |

### Ejemplos de Uso

#### 1. Autenticación
```bash
# Registro de usuario
curl -X POST http://localhost:8080/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepass123",
    "username": "johndoe"
  }'
```

#### 2. Crear Post
```bash
# Crear publicación
curl -X POST http://localhost:8080/api/posts \
  -H "X-User-Id: user123" \
  -H "X-Username: johndoe" \
  -H "Content-Type: application/json" \
  -d '{
    "imageUrl": "https://storage.url/image.jpg",
    "caption": "Mi primera publicación",
    "filter": "ups_logo"
  }'
```

#### 3. Aplicar Filtro CUDA
```bash
# Aplicar filtro a imagen
curl -X POST http://localhost:8080/api/filters/apply \
  -F "image=@photo.jpg" \
  -F "filter_name=gaussian"
```

---

## 🛠️ Desarrollo Local

### Prerrequisitos
- Java 21 JDK
- Maven 3.9+
- Docker (opcional)

### Compilación

```bash
# Limpiar y compilar
mvn clean package

# Compilar sin tests
mvn clean package -DskipTests
```

### Ejecución Local

```bash
# Método 1: Maven
mvn spring-boot:run

# Método 2: JAR
java -jar target/api-gateway-1.0.0.jar

# Método 3: Con perfil específico
java -jar target/api-gateway-1.0.0.jar --spring.profiles.active=dev
```

### PowerShell Scripts

```powershell
# Iniciar gateway
.\start-gateway.ps1

# Build y Docker
.\build-gateway-docker.ps1

# Testing
.\test-gateway.ps1
.\test-cuda-gateway.ps1
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
COPY --from=builder /app/target/api-gateway-*.jar app.jar
ENV SERVER_PORT=8080
EXPOSE 8080
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar --spring.profiles.active=${SPRING_PROFILES_ACTIVE}"]
```

### Docker Commands

```bash
# Build
docker build -t upsglam-api-gateway:latest .

# Run
docker run -d \
  --name api-gateway \
  -p 8080:8080 \
  -e SPRING_PROFILES_ACTIVE=docker \
  --network upsglam-network \
  upsglam-api-gateway:latest

# Logs
docker logs -f api-gateway

# Stop
docker stop api-gateway && docker rm api-gateway
```

### Docker Compose

```yaml
api-gateway:
  build:
    context: ./api-gateway
    dockerfile: Dockerfile
  container_name: upsglam-api-gateway
  ports:
    - "8080:8080"
  environment:
    - SERVER_PORT=8080
    - SPRING_PROFILES_ACTIVE=docker
    - JAVA_OPTS=-Xmx256m -Xms128m
  networks:
    - upsglam-network
  depends_on:
    - auth-service
    - post-service
    - cuda-backend
```

---

## 🔍 Monitoreo y Health Checks

### Actuator Endpoints

| Endpoint | Descripción | Público |
|----------|-------------|---------|
| `/actuator/health` | Estado del servicio | ✅ |
| `/actuator/info` | Información del servicio | ✅ |
| `/actuator/metrics` | Métricas de rendimiento | ✅ |

### Health Check
```bash
# Check gateway health
curl http://localhost:8080/actuator/health

# Response
{
  "status": "UP",
  "components": {
    "diskSpace": { "status": "UP" },
    "ping": { "status": "UP" }
  }
}
```

---

## 🧪 Testing

### Test de Conectividad

```bash
# 1. Test básico
curl http://localhost:8080/

# 2. Test Auth Service routing
curl http://localhost:8080/api/auth/health

# 3. Test Post Service routing
curl http://localhost:8080/api/posts/health

# 4. Test CUDA Service routing
curl http://localhost:8080/api/filters/health
```

### Scripts de Prueba

- `test-gateway.ps1`: Tests básicos del gateway
- `test-cuda-gateway.ps1`: Tests específicos de rutas CUDA

---

## 📊 Rendimiento

### Configuración de Memoria

```bash
# JVM Options recomendadas
JAVA_OPTS="-Xmx256m -Xms128m -XX:+UseG1GC"
```

### Tiempos de Respuesta Esperados
- Health check: < 50ms
- Routing overhead: < 10ms
- Total request (gateway + service): < 500ms

---

## 🔒 Seguridad

### CORS Configuration
```yaml
spring:
  cloud:
    gateway:
      globalcors:
        corsConfigurations:
          '[/**]':
            allowedOrigins: "*"
            allowedMethods: "*"
            allowedHeaders: "*"
```

### Headers de Seguridad
- `X-User-Id`: Identificador de usuario (propagado)
- `X-Username`: Nombre de usuario (propagado)
- `Authorization`: Token de autenticación (futuro)

---

## 🐛 Troubleshooting

### Problemas Comunes

#### 1. Gateway no puede conectar a servicios
```bash
# Verificar que los servicios estén arriba
docker ps | grep -E "auth-service|post-service|cuda-backend"

# Verificar red Docker
docker network inspect upsglam-network
```

#### 2. Timeout en requests
```yaml
# Aumentar timeouts en application.yml
spring:
  cloud:
    gateway:
      httpclient:
        connect-timeout: 5000
        response-timeout: 30s
```

#### 3. Error 503 Service Unavailable
- Verificar health de servicios downstream
- Verificar configuración de URIs en routes
- Revisar logs del gateway

---

## 📚 Referencias

- [Spring Cloud Gateway Documentation](https://docs.spring.io/spring-cloud-gateway/docs/current/reference/html/)
- [API Routes Documentation](./API-ROUTES.md)
- [Project Main README](../../README.md)

---

## 📝 Changelog

### Version 1.0.0
- ✅ Implementación inicial con Spring Cloud Gateway
- ✅ Rutas para Auth, Post y CUDA services
- ✅ Docker support
- ✅ Health checks y monitoring
- ✅ CORS configuration

---

## 👥 Autor

UPSGlam Development Team - Universidad Politécnica Salesiana
