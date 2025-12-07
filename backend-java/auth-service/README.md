# UPSGlam Auth Service

Microservicio de autenticación y gestión de usuarios con Firebase.

## Puerto
- **Auth Service**: `http://localhost:8082`

## Tecnologías
- Spring Boot 3.2.0 WebFlux (Reactive)
- Firebase Authentication
- Firebase Firestore
- Firebase Storage
- Java 21

## Endpoints

### Autenticación

#### POST /api/auth/register
Registrar nuevo usuario.

**Request:**
```json
{
  "email": "user@ups.edu.ec",
  "password": "12345678",
  "fullName": "Pepito Pérez",
  "username": "pepito"
}
```

**Response (200 OK):**
```json
{
  "user": {
    "id": "firebaseUid",
    "email": "user@ups.edu.ec",
    "username": "pepito",
    "fullName": "Pepito Pérez",
    "photoUrl": null,
    "bio": null
  },
  "token": {
    "idToken": "JWT_TOKEN",
    "refreshToken": null,
    "expiresIn": 3600
  }
}
```

#### POST /api/auth/login
Iniciar sesión (con email o username).

**Request:**
```json
{
  "identifier": "pepito",
  "password": "12345678"
}
```

**Response:** Mismo formato que register.

#### GET /api/auth/me
Obtener perfil del usuario autenticado.

**Headers:**
```
Authorization: Bearer <idToken>
```

**Response (200 OK):**
```json
{
  "id": "uid123",
  "email": "user@ups.edu.ec",
  "username": "pepito",
  "fullName": "Pepito Pérez",
  "photoUrl": "https://...",
  "bio": "Estudiante de la UPS"
}
```

### Perfil de Usuario

#### PATCH /api/users/me
Actualizar perfil de usuario.

**Headers:**
```
Authorization: Bearer <idToken>
```

**Request (todos los campos opcionales):**
```json
{
  "username": "pepito_cuda",
  "fullName": "Pepito Pérez",
  "bio": "Programando en PyCUDA 🤖"
}
```

**Response:** Datos del usuario actualizados.

## Configuración de Firebase

1. Crear proyecto en [Firebase Console](https://console.firebase.google.com/)

2. Habilitar Firebase Authentication:
   - Authentication > Sign-in method
   - Habilitar Email/Password

3. Crear Firestore Database:
   - Firestore Database > Create database
   - Modo: Production

4. Descargar credenciales:
   - Project Settings > Service Accounts
   - Generate new private key
   - Guardar como `firebase-credentials.json` en `src/main/resources/`

5. Configurar `application.yml`:
```yaml
firebase:
  credentials:
    path: classpath:firebase-credentials.json
  project-id: tu-project-id
  storage:
    bucket: tu-project.appspot.com
```

## Estructura de Firestore

### Colección: `users`
Documento: `users/{uid}`

```json
{
  "email": "user@ups.edu.ec",
  "username": "pepito",
  "fullName": "Pepito Pérez",
  "photoUrl": null,
  "bio": null,
  "createdAt": 1234567890
}
```

## Iniciar el servicio

```powershell
cd backend-java/auth-service
.\start-auth.ps1
```

## Probar endpoints

```powershell
.\test-auth.ps1
```

## Códigos de Error

- `400` - `VALIDATION_ERROR` - Datos inválidos
- `401` - `UNAUTHORIZED` - Token inválido o expirado
- `401` - `INVALID_CREDENTIALS` - Usuario o contraseña incorrectos
- `404` - `USER_NOT_FOUND` - Usuario no encontrado
- `409` - `EMAIL_ALREADY_IN_USE` - Email ya registrado
- `409` - `USERNAME_ALREADY_IN_USE` - Username ya en uso
- `500` - `INTERNAL_ERROR` - Error interno del servidor

## Integración con API Gateway

El API Gateway enrutará las peticiones:
```
/api/auth/** → auth-service:8082
```
