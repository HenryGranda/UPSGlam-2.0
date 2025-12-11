# API de Follows - UPSGlam

## Endpoints Implementados

### 1. Seguir a un Usuario
**POST** `/follows`

Permite que un usuario siga a otro usuario.

**Headers:**
```
Authorization: Bearer <idToken>
```

**Body:**
```json
{
  "targetUserId": "uid_del_usuario_a_seguir"
}
```

**Response Success (200):**
```json
{
  "success": true,
  "message": "Ahora sigues a este usuario",
  "isFollowing": true,
  "followersCount": 42
}
```

**Errores:**
- `400 SELF_FOLLOW_NOT_ALLOWED`: No puedes seguirte a ti mismo
- `409 ALREADY_FOLLOWING`: Ya estás siguiendo a este usuario
- `404 USER_NOT_FOUND`: Usuario no encontrado
- `401 UNAUTHORIZED`: Token inválido o faltante

---

### 2. Dejar de Seguir a un Usuario
**DELETE** `/follows/{userId}`

Permite que un usuario deje de seguir a otro.

**Headers:**
```
Authorization: Bearer <idToken>
```

**Path Parameters:**
- `userId`: ID del usuario a dejar de seguir

**Response Success (200):**
```json
{
  "success": true,
  "message": "Dejaste de seguir a este usuario",
  "isFollowing": false,
  "followersCount": 41
}
```

**Errores:**
- `404 FOLLOW_NOT_FOUND`: No estabas siguiendo a este usuario
- `404 USER_NOT_FOUND`: Usuario no encontrado
- `401 UNAUTHORIZED`: Token inválido o faltante

---

### 3. Obtener Estadísticas de Follows
**GET** `/follows/{userId}/stats`

Obtiene estadísticas de followers y following de un usuario.

**Headers:**
```
Authorization: Bearer <idToken>
```

**Path Parameters:**
- `userId`: ID del usuario

**Query Parameters:**
- `includeList` (opcional, default=false): Si es `true`, incluye las listas completas de seguidores y seguidos

**Response Success (200) - Sin listas:**
```json
{
  "userId": "uid123",
  "followersCount": 150,
  "followingCount": 75,
  "isFollowing": true
}
```

**Response Success (200) - Con listas (includeList=true):**
```json
{
  "userId": "uid123",
  "followersCount": 150,
  "followingCount": 75,
  "isFollowing": true,
  "followers": [
    {
      "id": "uid456",
      "username": "user1",
      "fullName": "User One",
      "photoUrl": "https://...",
      "bio": "Mi bio",
      "followersCount": 100,
      "followingCount": 50
    }
  ],
  "following": [
    {
      "id": "uid789",
      "username": "user2",
      "fullName": "User Two",
      "photoUrl": "https://...",
      "bio": "Otra bio",
      "followersCount": 200,
      "followingCount": 80
    }
  ]
}
```

**Notas:**
- `isFollowing` será `null` si consultas tus propias estadísticas
- Si consultas otro usuario, `isFollowing` indica si tú lo sigues

---

### 4. Obtener Lista de Seguidores
**GET** `/follows/{userId}/followers`

Obtiene la lista completa de seguidores de un usuario.

**Headers:**
```
Authorization: Bearer <idToken>
```

**Path Parameters:**
- `userId`: ID del usuario

**Response Success (200):**
```json
[
  {
    "id": "uid456",
    "username": "user1",
    "fullName": "User One",
    "photoUrl": "https://...",
    "bio": "Mi bio",
    "followersCount": 100,
    "followingCount": 50
  },
  {
    "id": "uid789",
    "username": "user2",
    "fullName": "User Two",
    "photoUrl": "https://...",
    "bio": "Otra bio",
    "followersCount": 200,
    "followingCount": 80
  }
]
```

---

### 5. Obtener Lista de Seguidos
**GET** `/follows/{userId}/following`

Obtiene la lista de usuarios que sigue un usuario.

**Headers:**
```
Authorization: Bearer <idToken>
```

**Path Parameters:**
- `userId`: ID del usuario

**Response Success (200):**
```json
[
  {
    "id": "uid789",
    "username": "user2",
    "fullName": "User Two",
    "photoUrl": "https://...",
    "bio": "Otra bio",
    "followersCount": 200,
    "followingCount": 80
  }
]
```

---

## Estructura de Base de Datos (Firestore)

### Colección: `users`
```
users/{userId}
  - email: string
  - username: string
  - fullName: string
  - photoUrl: string | null
  - bio: string | null
  - createdAt: timestamp
  - followersCount: number (contador de seguidores)
  - followingCount: number (contador de seguidos)
```

### Colección: `follows`
```
follows/{followerId}_{followedId}
  - followerUserId: string (quien sigue)
  - followedUserId: string (quien es seguido)
  - createdAt: timestamp
```

**Ejemplo:** Si el usuario `abc123` sigue al usuario `xyz789`, se crea:
```
follows/abc123_xyz789
  - followerUserId: "abc123"
  - followedUserId: "xyz789"
  - createdAt: 1702300000000
```

---

## Reglas de Negocio

1. ✅ Un usuario **NO puede seguirse a sí mismo**
2. ✅ Un usuario **NO puede seguir dos veces al mismo usuario** (se lanza error ALREADY_FOLLOWING)
3. ✅ Solo se puede hacer unfollow de usuarios que **actualmente sigues**
4. ✅ Los contadores se actualizan **automáticamente** al hacer follow/unfollow
5. ✅ Todas las operaciones requieren **autenticación** con token de Firebase

---

## Ejemplos de Uso

### Ejemplo 1: Seguir a un usuario
```bash
curl -X POST http://localhost:8080/api/auth/follows \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1..." \
  -H "Content-Type: application/json" \
  -d '{
    "targetUserId": "abc123xyz"
  }'
```

### Ejemplo 2: Dejar de seguir
```bash
curl -X DELETE http://localhost:8080/api/auth/follows/abc123xyz \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1..."
```

### Ejemplo 3: Ver estadísticas con listas
```bash
curl -X GET "http://localhost:8080/api/auth/follows/abc123xyz/stats?includeList=true" \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1..."
```

### Ejemplo 4: Ver solo seguidores
```bash
curl -X GET http://localhost:8080/api/auth/follows/abc123xyz/followers \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1..."
```

---

## Integración con API Gateway

**IMPORTANTE:** Todas las peticiones deben pasar por el API Gateway en puerto 8080:

```
POST   http://localhost:8080/api/auth/follows
DELETE http://localhost:8080/api/auth/follows/{userId}
GET    http://localhost:8080/api/auth/follows/{userId}/stats
GET    http://localhost:8080/api/auth/follows/{userId}/followers
GET    http://localhost:8080/api/auth/follows/{userId}/following
```

**Arquitectura:**
- API Gateway: `localhost:8080` - Punto de entrada único
- Auth Service: `localhost:8082` - Servicio interno (no acceder directamente)
- Post Service: `localhost:8081` - Servicio interno

---

## Próximos Pasos

### Recomendaciones para el Frontend (Flutter)

1. **Botón Follow/Unfollow en Perfil de Usuario**
2. **Pantalla de Seguidores**: Lista con avatar, username, botón follow
3. **Pantalla de Seguidos**: Similar a seguidores
4. **Feed Personalizado**: Mostrar solo posts de usuarios que sigues
5. **Contador Visual**: Mostrar `followersCount` y `followingCount` en perfil

### Para Post Service

Puedes integrar los follows en el feed:
- Crear endpoint `GET /feed/following` que filtre solo posts de usuarios seguidos
- Usar `FirebaseService` para obtener la lista de `following` del usuario actual
- Filtrar posts donde `userId IN followingList`

---

## Testing

Para probar la funcionalidad, puedes usar estos scripts de PowerShell:

```powershell
# En backend-java/auth-service/
.\start-auth.ps1

# Luego probar con curl o Postman
```
