# 🔐 Auth Service - Documentación Completa

## 📋 Descripción General

El **Auth Service** es el microservicio responsable de la **autenticación y gestión de usuarios** en UPSGlam. Integra Firebase Authentication para la autenticación y Firestore para almacenar perfiles de usuario.

**Puerto**: `8082`  
**Base Path**: `/api`  
**Tecnología**: Spring WebFlux (Reactivo)

---

## ⚙️ Configuración Rápida

### **Iniciar el Servidor**

```powershell
cd backend-java/auth-service
.\start-auth.ps1
```

El script `start-auth.ps1` configura automáticamente:
```powershell
$env:FIREBASE_API_KEY = "AIzaSyBYcnFxABxm3eyFpCD-nioQbZV1-NDzA5A"
$env:FIREBASE_PROJECT_ID = "upsglam-8c88f"
$env:FIREBASE_STORAGE_BUCKET = "upsglam-8c88f.appspot.com"
$env:FIREBASE_DATABASE_ID = "db-auth"
```

### **Ejecutar Tests Automáticos**

```powershell
cd docs
.\test-auth-flow.ps1
```

Este script ejecuta **8 tests completos**:
1. ✅ Registro + Login inicial
2. ✅ Obtener perfil (`GET /auth/me`)
3. ✅ Actualizar perfil (`PATCH /users/me`)
4. ✅ Verificar cambios
5. ✅ Login con email
6. ✅ Login con username
7. ✅ Credenciales incorrectas (test de seguridad)
8. ✅ Token inválido (test de seguridad)

**Resultado esperado**: Todos los tests pasan en ~15 segundos.

---

## 🚀 Flujo Completo del Usuario

### 1️⃣ **Registro de Usuario Nuevo**

Un nuevo usuario se registra en la aplicación proporcionando sus datos básicos.

#### **Endpoint**: `POST /auth/register`

**Request Body**:
```json
{
  "email": "usuario@ups.edu.ec",
  "password": "password123",
  "fullName": "Juan Pérez",
  "username": "juanperez"
}
```

**Validaciones**:
- ✅ Email debe ser válido y único
- ✅ Password mínimo 6 caracteres
- ✅ FullName entre 2-80 caracteres
- ✅ Username único, 3-20 caracteres (solo letras, números y guion bajo)

**Response** (201 Created):
```json
{
  "user": {
    "id": "firebase-uid-123",
    "email": "usuario@ups.edu.ec",
    "username": "juanperez",
    "fullName": "Juan Pérez",
    "photoUrl": null,
    "bio": null
  },
  "token": {
    "idToken": "eyJhbGciOiJSUzI1NiIs...",
    "refreshToken": "AEu4IL3F...",
    "expiresIn": 3600
  }
}
```

> ⚠️ **IMPORTANTE**: El token retornado por `/auth/register` es un **custom token** de Firebase.
> 
> - **Custom tokens** se generan en el servidor y deben ser intercambiados por un **ID token** en el cliente
> - Para usar endpoints autenticados inmediatamente después del registro, debes llamar a `/auth/login`
> - Los clientes móviles/web deben usar `signInWithCustomToken()` del SDK de Firebase para obtener el ID token
> - **No uses el custom token directamente** en el header `Authorization` - será rechazado con error 401

**Errores Posibles**:
- `400 BAD_REQUEST`: Datos inválidos
- `409 CONFLICT`: Email o username ya registrado

**Flujo Interno**:
1. Valida que el username no exista en Firestore
2. Crea usuario en Firebase Authentication
3. Guarda perfil en Firestore `users/{uid}`
4. Genera token de autenticación
5. Retorna usuario + token

---

### 2️⃣ **Inicio de Sesión**

Usuario existente inicia sesión con sus credenciales.

#### **Endpoint**: `POST /auth/login`

**Request Body**:
```json
{
  "identifier": "juanperez",
  "password": "password123"
}
```

**Nota**: `identifier` puede ser:
- ✉️ Email: `usuario@ups.edu.ec`
- 👤 Username: `juanperez`
- 📱 Teléfono: `+593999999999` (futuro)

**Response** (200 OK):
```json
{
  "user": {
    "id": "firebase-uid-123",
    "email": "usuario@ups.edu.ec",
    "username": "juanperez",
    "fullName": "Juan Pérez",
    "photoUrl": "https://storage.googleapis.com/avatars/firebase-uid-123.jpg",
    "bio": "Estudiante de Computación en la UPS 🚀"
  },
  "token": {
    "idToken": "eyJhbGciOiJSUzI1NiIs...",
    "refreshToken": "AEu4IL3F...",
    "expiresIn": 3600
  }
}
```

**Errores Posibles**:
- `400 BAD_REQUEST`: Datos faltantes
- `401 UNAUTHORIZED`: Credenciales incorrectas

**Flujo Interno**:
1. Determina si identifier es email o username
2. Si es username, busca el email correspondiente en Firestore
3. Autentica con Firebase Auth REST API
4. Obtiene perfil completo de Firestore
5. Retorna usuario + token real de Firebase

---

### 3️⃣ **Obtener Perfil Actual**

Usuario autenticado obtiene su información de perfil actualizada.

#### **Endpoint**: `GET /auth/me`

**Headers Requeridos**:
```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
```

**Response** (200 OK):
```json
{
  "id": "firebase-uid-123",
  "email": "usuario@ups.edu.ec",
  "username": "juanperez",
  "fullName": "Juan Pérez",
  "photoUrl": "https://storage.googleapis.com/avatars/firebase-uid-123.jpg",
  "bio": "Estudiante de Computación en la UPS 🚀"
}
```

**Errores Posibles**:
- `401 UNAUTHORIZED`: Token ausente o inválido
- `404 NOT_FOUND`: Usuario no existe en Firestore

**Casos de Uso**:
- 📱 Al abrir la app (splash screen)
- 🔄 Refrescar datos después de editar perfil
- ✅ Verificar sesión activa

---

### 4️⃣ **Actualizar Perfil**

Usuario modifica su información de perfil (username, nombre completo, bio).

#### **Endpoint**: `PATCH /users/me`

**Headers Requeridos**:
```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
Content-Type: application/json
```

**Request Body** (todos los campos opcionales):
```json
{
  "username": "juan_perez_ups",
  "fullName": "Juan Carlos Pérez",
  "bio": "Desarrollador Full Stack | UPS 🎓 | CUDA Enthusiast 💻"
}
```

**Validaciones**:
- Username: 3-20 caracteres (letras, números, guion bajo)
- FullName: 2-80 caracteres
- Bio: máximo 150 caracteres

**Response** (200 OK):
```json
{
  "id": "firebase-uid-123",
  "email": "usuario@ups.edu.ec",
  "username": "juan_perez_ups",
  "fullName": "Juan Carlos Pérez",
  "photoUrl": "https://storage.googleapis.com/avatars/firebase-uid-123.jpg",
  "bio": "Desarrollador Full Stack | UPS 🎓 | CUDA Enthusiast 💻"
}
```

**Errores Posibles**:
- `400 BAD_REQUEST`: Datos inválidos
- `401 UNAUTHORIZED`: Token inválido
- `409 CONFLICT`: Username ya está en uso

**Flujo Interno**:
1. Valida token y extrae userId
2. Si se actualiza username, verifica que no exista
3. Actualiza documento en Firestore `users/{uid}`
4. Retorna perfil actualizado

---

## 🔄 Flujo Completo de Usuario (Historia Típica)

### **Día 1 - Registro y Primera Publicación**

```mermaid
sequenceDiagram
    participant App
    participant Gateway
    participant AuthService
    participant Firebase
    participant Firestore

    Note over App: Usuario abre la app por primera vez
    
    App->>Gateway: POST /auth/register
    Gateway->>AuthService: Forward request
    AuthService->>Firestore: Check username exists
    Firestore-->>AuthService: Username available
    AuthService->>Firebase: Create user
    Firebase-->>AuthService: User created (uid)
    AuthService->>Firestore: Save user profile
    AuthService-->>App: {user, token}
    
    Note over App: Usuario recibe token y navega al feed
    
    App->>Gateway: GET /auth/me (con token)
    Gateway->>AuthService: Verify token + get profile
    AuthService->>Firestore: Get user by uid
    Firestore-->>AuthService: User profile
    AuthService-->>App: User data
    
    Note over App: Usuario navega a su perfil
```

### **Día 2 - Login y Actualización de Perfil**

```mermaid
sequenceDiagram
    participant App
    participant Gateway
    participant AuthService
    participant Firebase
    participant Firestore

    Note over App: Usuario abre la app (sesión expirada)
    
    App->>Gateway: POST /auth/login
    Gateway->>AuthService: {identifier, password}
    AuthService->>Firestore: Get email by username
    Firestore-->>AuthService: email found
    AuthService->>Firebase: Authenticate user
    Firebase-->>AuthService: Auth successful
    AuthService->>Firestore: Get full profile
    Firestore-->>AuthService: User data
    AuthService-->>App: {user, token}
    
    Note over App: Usuario edita su bio
    
    App->>Gateway: PATCH /users/me {bio: "..."}
    Gateway->>AuthService: Update profile
    AuthService->>Firestore: Update user doc
    Firestore-->>AuthService: Updated
    AuthService-->>App: Updated user
    
    Note over App: Usuario refresca perfil
    
    App->>Gateway: GET /auth/me
    Gateway->>AuthService: Get current user
    AuthService->>Firestore: Fetch user
    AuthService-->>App: User with new bio
```

---

## 📊 Modelos de Datos

### **User (Firestore Document)**

Colección: `users/{uid}`

```json
{
  "email": "usuario@ups.edu.ec",
  "username": "juanperez",
  "fullName": "Juan Pérez",
  "photoUrl": "https://storage.googleapis.com/avatars/uid-123.jpg",
  "bio": "Estudiante de la UPS 🚀",
  "createdAt": "2025-12-08T10:00:00Z",
  "updatedAt": "2025-12-08T15:30:00Z"
}
```

### **Firebase Authentication**

Firebase Auth almacena:
- Email
- Password (hash)
- UID (identificador único)
- Email verificado
- Fecha de creación

---

## 🔒 Autenticación y Seguridad

### **Token JWT (Firebase ID Token)**

El token JWT incluye:
```json
{
  "iss": "https://securetoken.google.com/upsglam-8c88f",
  "aud": "upsglam-8c88f",
  "auth_time": 1733656800,
  "user_id": "firebase-uid-123",
  "sub": "firebase-uid-123",
  "iat": 1733656800,
  "exp": 1733660400,
  "email": "usuario@ups.edu.ec",
  "email_verified": false
}
```

**Expiración**: 1 hora (3600 segundos)

### **Headers de Autorización**

Todos los endpoints protegidos requieren:
```http
Authorization: Bearer {idToken}
```

El API Gateway valida el token antes de reenviar al servicio.

---

## 🧪 Testing del Flujo Completo

### **Script de Test PowerShell**

```powershell
# Variables
$BASE_URL = "http://localhost:8080"
$EMAIL = "test$(Get-Random)@ups.edu.ec"
$PASSWORD = "test123456"
$USERNAME = "testuser$(Get-Random)"

# 1. Registro
Write-Host "🔵 Registrando usuario..." -ForegroundColor Blue
$registerBody = @{
    email = $EMAIL
    password = $PASSWORD
    fullName = "Usuario de Prueba"
    username = $USERNAME
} | ConvertTo-Json

$registerResponse = Invoke-RestMethod -Uri "$BASE_URL/auth/register" `
    -Method POST `
    -ContentType "application/json" `
    -Body $registerBody

$TOKEN = $registerResponse.token.idToken
Write-Host "✅ Usuario registrado: $($registerResponse.user.username)" -ForegroundColor Green

# 2. Obtener perfil
Write-Host "`n🔵 Obteniendo perfil..." -ForegroundColor Blue
$headers = @{ Authorization = "Bearer $TOKEN" }
$meResponse = Invoke-RestMethod -Uri "$BASE_URL/auth/me" `
    -Method GET `
    -Headers $headers

Write-Host "✅ Perfil obtenido: $($meResponse.fullName)" -ForegroundColor Green

# 3. Actualizar perfil
Write-Host "`n🔵 Actualizando perfil..." -ForegroundColor Blue
$updateBody = @{
    bio = "¡Probando la API de UPSGlam! 🚀"
} | ConvertTo-Json

$updateResponse = Invoke-RestMethod -Uri "$BASE_URL/users/me" `
    -Method PATCH `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $updateBody

Write-Host "✅ Perfil actualizado. Bio: $($updateResponse.bio)" -ForegroundColor Green

# 4. Login
Write-Host "`n🔵 Probando login..." -ForegroundColor Blue
$loginBody = @{
    identifier = $USERNAME
    password = $PASSWORD
} | ConvertTo-Json

$loginResponse = Invoke-RestMethod -Uri "$BASE_URL/auth/login" `
    -Method POST `
    -ContentType "application/json" `
    -Body $loginBody

Write-Host "✅ Login exitoso: $($loginResponse.user.email)" -ForegroundColor Green

# Resumen
Write-Host "`n✅ TODAS LAS PRUEBAS PASARON" -ForegroundColor Green
Write-Host "Usuario: $USERNAME" -ForegroundColor Cyan
Write-Host "Email: $EMAIL" -ForegroundColor Cyan
Write-Host "Token: $($TOKEN.Substring(0, 50))..." -ForegroundColor Cyan
```

### **Ejecutar Tests**

```powershell
# Iniciar auth-service
cd backend-java/auth-service
.\start-auth.ps1

# En otra terminal, ejecutar tests
.\test-auth.ps1
```

---

## 🐛 Troubleshooting

### **Error: Firebase credentials not found**

```bash
# Solución: Configurar variable de entorno
$env:FIREBASE_CREDENTIALS_PATH = "c:/path/to/firebase-credentials.json"
```

### **Error: Username already in use**

El username debe ser único en todo el sistema. Usar otro.

### **Error: Token expired**

Los tokens expiran en 1 hora. Hacer login nuevamente.

### **Error: API key not valid (400 Bad Request al hacer login)**

**Síntoma**: Registro exitoso, usuario guardado en Firestore, pero login falla con:
```
400 Bad Request: API key not valid. Please pass a valid API key.
```

**Causa**: El API Key configurado NO es el correcto para Firebase Authentication REST API.

**Solución**:
1. Ve a Firebase Console: https://console.firebase.google.com/project/upsglam-8c88f/settings/general
2. En la sección **"Your apps"**, busca **"Web API Key"**
3. Copia el key completo
4. Actualiza en `start-auth.ps1`:
   ```powershell
   $env:FIREBASE_API_KEY = "AIzaSyBYcnFxABxm3eyFpCD-nioQbZV1-NDzA5A"
   ```
5. Reinicia el servidor con `.\start-auth.ps1`

**IMPORTANTE**: 
- Usa el **Web API Key** (sin restricciones) del Firebase Config
- NO uses "Server Key" ni "Restricted Key"
- El Web API Key es el único que funciona con `identitytoolkit.googleapis.com`

### **Error: Usuarios se registran pero no pueden hacer login (400 Bad Request)**

**Síntoma**: Registro exitoso (201 Created) pero login falla con `400 Bad Request from Firebase API`.

**Causa**: Firebase Admin SDK solo soporta el database Firestore `(default)`. Si configuraste un database personalizado (como `db-auth`), los usuarios se crean en Firebase Authentication pero NO se guardan en Firestore.

**Solución**:
1. **Opción 1 (Recomendada)**: Usar el database `(default)` de Firestore
   - En Firebase Console, verifica que tu colección `users` esté en el database `(default)`
   - El código ya está configurado para esto

2. **Opción 2**: Migrar datos de `db-auth` a `(default)` si ya tienes usuarios guardados
   ```bash
   # Exportar desde db-auth
   gcloud firestore export gs://bucket/backup --database=db-auth
   
   # Importar a (default)
   gcloud firestore import gs://bucket/backup --database=(default)
   ```

**Verificación**:
```bash
# Los logs deben mostrar:
INFO - Usuario creado en Firebase Auth: abc123
INFO - Usuario guardado en Firestore: abc123  # <-- Este log debe aparecer
INFO - Usuario registrado exitosamente: username
```

### **Error: CORS blocked**

Verificar configuración de CORS en API Gateway:
```yaml
spring:
  cloud:
    gateway:
      globalcors:
        cors-configurations:
          '[/**]':
            allowed-origins: "*"
            allowed-methods: "*"
            allowed-headers: "*"
```

---

## 📈 Métricas y Logs

### **Health Check**

```bash
GET http://localhost:8082/actuator/health
```

Response:
```json
{
  "status": "UP"
}
```

### **Logs Importantes**

```
INFO  - Registrando usuario: usuario@ups.edu.ec
INFO  - Usuario registrado exitosamente: juanperez
INFO  - Intentando login: juanperez
INFO  - Login exitoso para usuario: firebase-uid-123
INFO  - Actualizando perfil para usuario: firebase-uid-123
```

---

## 🚀 Integración con Otros Servicios

### **API Gateway**

El API Gateway enruta las peticiones:

```yaml
- id: auth-service-register
  uri: http://localhost:8082
  predicates:
    - Path=/auth/register
    - Method=POST

- id: auth-service-login
  uri: http://localhost:8082
  predicates:
    - Path=/auth/login
    - Method=POST

- id: auth-service-me
  uri: http://localhost:8082
  predicates:
    - Path=/auth/me
    - Method=GET

- id: auth-service-users
  uri: http://localhost:8082
  predicates:
    - Path=/users/**
```

### **Post Service**

Post Service usa el token para identificar al usuario:

```java
// En PostHandler
String userId = request.headers().firstHeader("X-User-Id");
String username = request.headers().firstHeader("X-Username");
```

El Gateway extrae estos datos del token JWT y los inyecta como headers.

---

## 📚 Endpoints Resumen

| Método | Endpoint | Descripción | Auth |
|--------|----------|-------------|------|
| POST | `/auth/register` | Registrar nuevo usuario | ❌ |
| POST | `/auth/login` | Iniciar sesión | ❌ |
| GET | `/auth/me` | Obtener perfil actual | ✅ |
| PATCH | `/users/me` | Actualizar perfil | ✅ |
| GET | `/actuator/health` | Health check | ❌ |

---

## 🎯 Próximos Pasos

Después de probar el auth-service, continuar con:

1. **Post Service** - Crear publicaciones con filtros CUDA
2. **Like System** - Dar like a posts
3. **Comment System** - Comentar en posts
4. **Mobile App** - Integrar Flutter/React Native

---

## 📞 Soporte

**Desarrollador**: Anthony  
**Proyecto**: UPSGlam 2.0  
**Fecha**: Diciembre 2025  
**Universidad**: Universidad Politécnica Salesiana
