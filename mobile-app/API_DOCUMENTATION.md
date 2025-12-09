# UPSGlam 2.0 - API Documentation for Mobile App

> 📱 Documentación completa para integrar la aplicación móvil Flutter con el backend de UPSGlam 2.0

**Última actualización:** Diciembre 9, 2025  
**Versión de API:** 2.0  
**Fecha límite de entrega:** Diciembre 10, 2025

---

## 📋 Tabla de Contenidos

1. [Configuración General](#configuración-general)
2. [Autenticación (Auth Service)](#autenticación-auth-service)
3. [Posts y Feed (Post Service)](#posts-y-feed-post-service)
4. [Imágenes y Filtros CUDA](#imágenes-y-filtros-cuda)
5. [Likes y Comentarios](#likes-y-comentarios)
6. [Modelos de Datos](#modelos-de-datos)
7. [Códigos de Error](#códigos-de-error)
8. [Flujos Completos](#flujos-completos)

---

## 🌐 Configuración General

### Base URL
```
API Gateway: http://localhost:8080/api
```

**IMPORTANTE:** Todas las peticiones DEBEN hacerse al API Gateway en el puerto **8080**. El gateway se encarga de enrutar internamente a los microservicios correctos.

### Arquitectura

```
Mobile App (Flutter)
        ↓
API Gateway (8080)
        ↓
   ┌────┴────┬────────────┐
   ↓         ↓            ↓
Auth      Post       PyCUDA
(8082)    (8081)     (5000)
```

### Headers Requeridos

Para endpoints protegidos (la mayoría):

```dart
{
  'Authorization': 'Bearer $idToken',
  'X-User-Id': '$userId',
  'Content-Type': 'application/json'
}
```

### CORS
El gateway tiene CORS habilitado para todos los orígenes, métodos y headers.

---

## 🔐 Autenticación (Auth Service)

### 1. Registro de Usuario

**Endpoint:** `POST /api/auth/register`

**Request Body:**
```json
{
  "email": "usuario@ups.edu.ec",
  "password": "password123",
  "username": "usuario123",
  "fullName": "Juan Pérez"
}
```

**Response (201):**
```json
{
  "user": {
    "id": "FsRqByuHJgXy2M08GvnNDI4htHm2",
    "email": "usuario@ups.edu.ec",
    "username": "usuario123",
    "fullName": "Juan Pérez",
    "bio": null,
    "photoUrl": null,
    "createdAt": "2025-12-09T18:30:00.000Z"
  },
  "token": {
    "idToken": "eyJhbGciOiJSUzI1NiIsImtpZCI6IjM4...",
    "refreshToken": "AGEh...",
    "expiresIn": "3600"
  }
}
```

**Validaciones:**
- Email: Formato válido y único
- Password: Mínimo 6 caracteres
- Username: Único, alfanumérico, guiones permitidos
- FullName: Requerido

**Flujo después del registro:**
1. Guardar `user.id` como `userId`
2. Guardar `token.idToken` como `idToken`
3. **IMPORTANTE:** Después del registro, esperar 3 segundos antes de hacer login para que Firebase active completamente el usuario
4. Hacer login inmediatamente para obtener el ID token válido

### 2. Login

**Endpoint:** `POST /api/auth/login`

**Request Body:**
```json
{
  "identifier": "usuario@ups.edu.ec",
  "password": "password123"
}
```

> **Nota:** `identifier` puede ser email o username

**Response (200):**
```json
{
  "user": {
    "id": "FsRqByuHJgXy2M08GvnNDI4htHm2",
    "username": "usuario123",
    "email": "usuario@ups.edu.ec",
    "fullName": "Juan Pérez",
    "bio": "Estudiante UPS",
    "photoUrl": "https://...",
    "createdAt": "2025-12-09T18:30:00.000Z"
  },
  "token": {
    "idToken": "eyJhbGciOiJSUzI1NiIsImtpZCI6IjM4...",
    "refreshToken": "AGEh...",
    "expiresIn": "3600"
  }
}
```

**Errores:**
- `401`: Credenciales incorrectas
- `404`: Usuario no encontrado

### 3. Obtener Perfil Actual

**Endpoint:** `GET /api/auth/me`

**Headers:**
```
Authorization: Bearer {idToken}
```

**Response (200):**
```json
{
  "id": "FsRqByuHJgXy2M08GvnNDI4htHm2",
  "username": "usuario123",
  "email": "usuario@ups.edu.ec",
  "fullName": "Juan Pérez",
  "bio": "Estudiante UPS | Full Stack Dev",
  "photoUrl": "https://...",
  "createdAt": "2025-12-09T18:30:00.000Z"
}
```

### 4. Actualizar Perfil

**Endpoint:** `PATCH /api/users/me`

**Headers:**
```
Authorization: Bearer {idToken}
Content-Type: application/json
```

**Request Body (todos opcionales):**
```json
{
  "username": "nuevo_username",
  "fullName": "Nuevo Nombre",
  "bio": "Nueva biografía",
  "photoUrl": "https://nueva-foto.jpg"
}
```

**Response (200):**
```json
{
  "id": "FsRqByuHJgXy2M08GvnNDI4htHm2",
  "username": "nuevo_username",
  "email": "usuario@ups.edu.ec",
  "fullName": "Nuevo Nombre",
  "bio": "Nueva biografía",
  "photoUrl": "https://nueva-foto.jpg",
  "updatedAt": "2025-12-09T18:35:00.000Z"
}
```

---

## 📸 Posts y Feed (Post Service)

### 5. Subir Imagen Simple (Sin Filtro)

**Endpoint:** `POST /api/images/upload`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
Content-Type: multipart/form-data
```

**Request (Multipart):**
```
image: [archivo de imagen JPEG/PNG]
```

**Response (200):**
```json
{
  "imageId": "FsRqByuHJgXy2M08GvnNDI4htHm2-1733789123456.jpg",
  "imageUrl": "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/posts/FsRqByuHJgXy2M08GvnNDI4htHm2-1733789123456.jpg"
}
```

**Uso:**
- Para subir imágenes sin aplicar filtros CUDA
- La imagen se sube directamente a Supabase en la carpeta `posts/`
- Usar `imageUrl` al crear el post

### 6. Previsualizar Imagen con Filtro CUDA

**Endpoint:** `POST /api/images/preview`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
Content-Type: multipart/form-data
```

**Request (Multipart):**
```
image: [archivo de imagen JPEG/PNG]
filter: "gaussian"
```

**Filtros disponibles:**
- `gaussian` - Desenfoque gaussiano (suave)
- `box_blur` - Desenfoque de caja
- `prewitt` - Detección de bordes Prewitt
- `laplacian` - Detección de bordes Laplacian
- `ups_logo` - Overlay del logo UPS
- `ups_color` - Filtro de colores UPS

**Response (200):**
```json
{
  "tempImageId": "temp_FsRqByuHJgXy2M08GvnNDI4htHm2_1733789123456.jpg",
  "imageUrl": "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/temp/temp_FsRqByuHJgXy2M08GvnNDI4htHm2_1733789123456.jpg",
  "filter": "gaussian"
}
```

**Flujo interno:**
1. Post-service recibe la imagen
2. **Post-service → PyCUDA (puerto 5000)** - Aplica filtro con GPU
3. PyCUDA devuelve imagen procesada
4. Post-service sube a Supabase en carpeta `temp/`
5. Devuelve URL temporal para previsualización

**Uso en la app:**
1. Usuario selecciona imagen
2. Usuario selecciona filtro
3. App envía a `/images/preview` con filtro
4. Mostrar `imageUrl` en la UI para que usuario vea el resultado
5. Usuario confirma o cambia de filtro
6. Al crear post, usar el `imageUrl` devuelto

### 7. Crear Post

**Endpoint:** `POST /api/posts`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
Content-Type: application/json
```

**Request Body:**

**Opción A - Con imagen ya subida (sin filtro):**
```json
{
  "mediaUrl": "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/posts/imagen.jpg",
  "caption": "Mi primer post! #UPSGlam",
  "filter": null,
  "username": "usuario123"
}
```

**Opción B - Con imagen de previsualización (con filtro):**
```json
{
  "tempImageId": "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/temp/temp_imagen.jpg",
  "caption": "Post con filtro CUDA! #UPSGlam",
  "filter": "gaussian",
  "username": "usuario123"
}
```

**Response (201):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "userId": "FsRqByuHJgXy2M08GvnNDI4htHm2",
  "username": "usuario123",
  "caption": "Mi primer post! #UPSGlam",
  "imageUrl": "https://opohishcukgkrkfdsgoa.supabase.co/storage/v1/object/public/upsglam/posts/imagen.jpg",
  "filter": null,
  "likesCount": 0,
  "commentsCount": 0,
  "isLikedByMe": false,
  "createdAt": "2025-12-09T18:40:00.000Z"
}
```

**Flujo interno con tempImageId:**
1. Post-service recibe `tempImageId`
2. Mueve imagen de `temp/` a `posts/` en Supabase
3. Guarda post en Firestore con URL final
4. Devuelve post completo

### 8. Obtener Feed

**Endpoint:** `GET /api/feed?limit=20&page=0`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
```

**Query Parameters:**
- `limit` (opcional): Número de posts por página (default: 20)
- `page` (opcional): Número de página (default: 0)

**Response (200):**
```json
{
  "posts": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "userId": "FsRqByuHJgXy2M08GvnNDI4htHm2",
      "username": "usuario123",
      "userPhotoUrl": "https://...",
      "caption": "Post de prueba",
      "imageUrl": "https://...",
      "filter": "gaussian",
      "likesCount": 5,
      "commentsCount": 3,
      "isLikedByMe": true,
      "createdAt": "2025-12-09T18:40:00.000Z"
    },
    ...
  ],
  "page": 0,
  "limit": 20,
  "total": 50
}
```

**Uso:**
- Mostrar en la pantalla principal del feed
- Implementar infinite scroll con paginación
- `isLikedByMe` indica si el usuario actual dio like

### 9. Obtener Post por ID

**Endpoint:** `GET /api/posts/{postId}`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
```

**Response (200):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "userId": "FsRqByuHJgXy2M08GvnNDI4htHm2",
  "username": "usuario123",
  "userPhotoUrl": "https://...",
  "caption": "Post de prueba",
  "imageUrl": "https://...",
  "filter": "gaussian",
  "likesCount": 5,
  "commentsCount": 3,
  "isLikedByMe": true,
  "createdAt": "2025-12-09T18:40:00.000Z"
}
```

### 10. Obtener Posts de un Usuario

**Endpoint:** `GET /api/posts/user/{userId}`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
```

**Response (200):**
```json
{
  "posts": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "userId": "FsRqByuHJgXy2M08GvnNDI4htHm2",
      "username": "usuario123",
      "caption": "Post de prueba",
      "imageUrl": "https://...",
      "filter": "gaussian",
      "likesCount": 5,
      "commentsCount": 3,
      "isLikedByMe": true,
      "createdAt": "2025-12-09T18:40:00.000Z"
    },
    ...
  ],
  "total": 10
}
```

**Uso:**
- Para mostrar perfil de usuario con sus posts
- Grid view de posts del usuario

### 11. Actualizar Caption

**Endpoint:** `PATCH /api/posts/{postId}/caption`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
Content-Type: application/json
```

**Request Body:**
```json
{
  "caption": "Nuevo caption actualizado"
}
```

**Response (200):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "caption": "Nuevo caption actualizado",
  "updatedAt": "2025-12-09T18:45:00.000Z"
}
```

**Restricción:**
- Solo el dueño del post puede actualizar el caption

### 12. Eliminar Post

**Endpoint:** `DELETE /api/posts/{postId}`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
```

**Response (204):**
```
No Content
```

**Restricción:**
- Solo el dueño del post puede eliminarlo
- Se eliminan en cascada todos los likes y comentarios

---

## ❤️ Likes y Comentarios

### 13. Dar Like a un Post

**Endpoint:** `POST /api/posts/{postId}/likes`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
```

**Response (201):**
```json
{
  "id": "like-id-123",
  "postId": "550e8400-e29b-41d4-a716-446655440000",
  "userId": "FsRqByuHJgXy2M08GvnNDI4htHm2",
  "createdAt": "2025-12-09T18:50:00.000Z"
}
```

**Uso:**
- Doble tap en la imagen o botón de corazón
- Si ya tiene like, mostrar error o ignorar

### 14. Quitar Like

**Endpoint:** `DELETE /api/posts/{postId}/likes`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
```

**Response (204):**
```
No Content
```

### 15. Obtener Likes de un Post

**Endpoint:** `GET /api/posts/{postId}/likes`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
```

**Response (200):**
```json
{
  "likes": [
    {
      "id": "like-id-123",
      "userId": "user-id-1",
      "username": "usuario1",
      "userPhotoUrl": "https://...",
      "createdAt": "2025-12-09T18:50:00.000Z"
    },
    ...
  ],
  "total": 15
}
```

**Uso:**
- Mostrar lista de usuarios que dieron like
- Al hacer tap en el contador de likes

### 16. Crear Comentario

**Endpoint:** `POST /api/posts/{postId}/comments`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "Excelente post! Me encanta 🔥"
}
```

**Response (201):**
```json
{
  "id": "comment-id-456",
  "postId": "550e8400-e29b-41d4-a716-446655440000",
  "userId": "FsRqByuHJgXy2M08GvnNDI4htHm2",
  "username": "usuario123",
  "userPhotoUrl": "https://...",
  "text": "Excelente post! Me encanta 🔥",
  "createdAt": "2025-12-09T18:55:00.000Z"
}
```

**Validaciones:**
- `text`: Requerido, máximo 500 caracteres

### 17. Obtener Comentarios de un Post

**Endpoint:** `GET /api/posts/{postId}/comments`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
```

**Response (200):**
```json
{
  "comments": [
    {
      "id": "comment-id-456",
      "postId": "550e8400-e29b-41d4-a716-446655440000",
      "userId": "FsRqByuHJgXy2M08GvnNDI4htHm2",
      "username": "usuario123",
      "userPhotoUrl": "https://...",
      "text": "Excelente post!",
      "createdAt": "2025-12-09T18:55:00.000Z"
    },
    ...
  ],
  "total": 8
}
```

**Uso:**
- Mostrar al abrir pantalla de comentarios
- Ordenados por fecha (más recientes primero)

### 18. Obtener Comentarios de un Usuario

**Endpoint:** `GET /api/users/{userId}/comments`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
```

**Response (200):**
```json
{
  "comments": [
    {
      "id": "comment-id-456",
      "postId": "550e8400-e29b-41d4-a716-446655440000",
      "userId": "FsRqByuHJgXy2M08GvnNDI4htHm2",
      "username": "usuario123",
      "text": "Excelente post!",
      "createdAt": "2025-12-09T18:55:00.000Z"
    },
    ...
  ],
  "total": 25
}
```

**Uso:**
- Ver todos los comentarios que un usuario ha hecho
- Historial de actividad del usuario

### 19. Eliminar Comentario

**Endpoint:** `DELETE /api/posts/{postId}/comments/{commentId}`

**Headers:**
```
Authorization: Bearer {idToken}
X-User-Id: {userId}
```

**Response (204):**
```
No Content
```

**Restricción:**
- Solo el dueño del comentario puede eliminarlo
- O el dueño del post puede eliminar cualquier comentario

---

## 🎨 Imágenes y Filtros CUDA

### 20. Listar Filtros Disponibles

**Endpoint:** `GET /api/filters/list`

**No requiere autenticación**

**Response (200):**
```json
{
  "filters": [
    {
      "name": "gaussian",
      "displayName": "Gaussian Blur",
      "description": "Suavizado gaussiano para desenfocar la imagen",
      "category": "blur"
    },
    {
      "name": "box_blur",
      "displayName": "Box Blur",
      "description": "Desenfoque de caja uniforme",
      "category": "blur"
    },
    {
      "name": "prewitt",
      "displayName": "Prewitt Edge",
      "description": "Detección de bordes con operador Prewitt",
      "category": "edge"
    },
    {
      "name": "laplacian",
      "displayName": "Laplacian Edge",
      "description": "Detección de bordes con operador Laplaciano",
      "category": "edge"
    },
    {
      "name": "ups_logo",
      "displayName": "UPS Logo",
      "description": "Overlay del logo de la Universidad",
      "category": "creative"
    },
    {
      "name": "ups_color",
      "displayName": "UPS Colors",
      "description": "Aplica los colores institucionales de la UPS",
      "category": "creative"
    }
  ]
}
```

**Uso:**
- Mostrar lista de filtros en selector
- Categorizar filtros por tipo (blur, edge, creative)

---

## 📦 Modelos de Datos

### User Model
```dart
class User {
  final String id;
  final String username;
  final String email;
  final String fullName;
  final String? bio;
  final String? photoUrl;
  final DateTime createdAt;
  final DateTime? updatedAt;
}
```

### Post Model
```dart
class Post {
  final String id;
  final String userId;
  final String username;
  final String? userPhotoUrl;
  final String caption;
  final String imageUrl;
  final String? filter;
  final int likesCount;
  final int commentsCount;
  final bool isLikedByMe;
  final DateTime createdAt;
  final DateTime? updatedAt;
}
```

### Comment Model
```dart
class Comment {
  final String id;
  final String postId;
  final String userId;
  final String username;
  final String? userPhotoUrl;
  final String text;
  final DateTime createdAt;
}
```

### Like Model
```dart
class Like {
  final String id;
  final String postId;
  final String userId;
  final String username;
  final String? userPhotoUrl;
  final DateTime createdAt;
}
```

### Token Model
```dart
class AuthToken {
  final String idToken;
  final String refreshToken;
  final String expiresIn;
}
```

---

## ⚠️ Códigos de Error

### Errores de Autenticación
- `400` - Bad Request: Datos inválidos o faltantes
- `401` - Unauthorized: Token inválido o expirado
- `403` - Forbidden: No tiene permisos para esta acción
- `404` - Not Found: Usuario no encontrado
- `409` - Conflict: Email o username ya existe

### Errores de Posts
- `400` - Bad Request: Datos inválidos (imagen, caption, etc.)
- `401` - Unauthorized: Token requerido
- `403` - Forbidden: No es el dueño del recurso
- `404` - Not Found: Post no encontrado
- `413` - Payload Too Large: Imagen muy grande (>10MB)

### Errores de PyCUDA
- `400` - Invalid Filter: Filtro no válido o imagen corrupta
- `503` - Service Unavailable: PyCUDA service no disponible o error en GPU

### Ejemplo de Response de Error
```json
{
  "error": "INVALID_CREDENTIALS",
  "message": "Email o contraseña incorrectos",
  "timestamp": "2025-12-09T18:30:00.000Z"
}
```

---

## 🔄 Flujos Completos

### Flujo 1: Registro e Inicio de Sesión

```dart
// 1. Registro
final registerResponse = await http.post(
  Uri.parse('$baseUrl/auth/register'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'email': email,
    'password': password,
    'username': username,
    'fullName': fullName,
  }),
);

if (registerResponse.statusCode == 201) {
  final data = jsonDecode(registerResponse.body);
  
  // IMPORTANTE: Esperar 3 segundos para activación de Firebase
  await Future.delayed(Duration(seconds: 3));
  
  // 2. Login inmediato después del registro
  final loginResponse = await http.post(
    Uri.parse('$baseUrl/auth/login'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'identifier': email,
      'password': password,
    }),
  );
  
  if (loginResponse.statusCode == 200) {
    final loginData = jsonDecode(loginResponse.body);
    
    // 3. Guardar tokens y datos de usuario
    await saveToken(loginData['token']['idToken']);
    await saveUserId(loginData['user']['id']);
    await saveUser(loginData['user']);
    
    // 4. Navegar a home
    Navigator.pushReplacement(context, HomePage());
  }
}
```

### Flujo 2: Crear Post sin Filtro

```dart
// 1. Seleccionar imagen
final XFile? image = await ImagePicker().pickImage(source: ImageSource.gallery);

// 2. Subir imagen
var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/images/upload'));
request.headers['Authorization'] = 'Bearer $idToken';
request.headers['X-User-Id'] = userId;
request.files.add(await http.MultipartFile.fromPath('image', image!.path));

var uploadResponse = await request.send();
var uploadData = jsonDecode(await uploadResponse.stream.bytesToString());

final imageUrl = uploadData['imageUrl'];

// 3. Crear post
final postResponse = await http.post(
  Uri.parse('$baseUrl/posts'),
  headers: {
    'Authorization': 'Bearer $idToken',
    'X-User-Id': userId,
    'Content-Type': 'application/json',
  },
  body: jsonEncode({
    'mediaUrl': imageUrl,
    'caption': caption,
    'filter': null,
    'username': username,
  }),
);

// 4. Actualizar UI con nuevo post
if (postResponse.statusCode == 201) {
  final post = Post.fromJson(jsonDecode(postResponse.body));
  // Agregar post al feed
}
```

### Flujo 3: Crear Post con Filtro CUDA

```dart
// 1. Seleccionar imagen
final XFile? image = await ImagePicker().pickImage(source: ImageSource.gallery);

// 2. Usuario selecciona filtro
String selectedFilter = 'gaussian'; // De una lista de filtros

// 3. Previsualizar con filtro
var previewRequest = http.MultipartRequest('POST', Uri.parse('$baseUrl/images/preview'));
previewRequest.headers['Authorization'] = 'Bearer $idToken';
previewRequest.headers['X-User-Id'] = userId;
previewRequest.files.add(await http.MultipartFile.fromPath('image', image!.path));
previewRequest.fields['filter'] = selectedFilter;

var previewResponse = await previewRequest.send();
var previewData = jsonDecode(await previewResponse.stream.bytesToString());

final tempImageUrl = previewData['imageUrl'];

// 4. Mostrar previsualización al usuario
showDialog(
  context: context,
  builder: (context) => PreviewDialog(
    imageUrl: tempImageUrl,
    onConfirm: () async {
      // 5. Usuario confirma, crear post
      final postResponse = await http.post(
        Uri.parse('$baseUrl/posts'),
        headers: {
          'Authorization': 'Bearer $idToken',
          'X-User-Id': userId,
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'tempImageId': tempImageUrl,
          'caption': caption,
          'filter': selectedFilter,
          'username': username,
        }),
      );
      
      if (postResponse.statusCode == 201) {
        // Post creado con filtro CUDA aplicado
        Navigator.pop(context);
        Navigator.pushReplacement(context, HomePage());
      }
    },
    onCancel: () {
      // Usuario puede elegir otro filtro
      Navigator.pop(context);
    },
  ),
);
```

### Flujo 4: Interacción con Post (Like y Comentario)

```dart
// 1. Dar like
Future<void> likePost(String postId) async {
  final response = await http.post(
    Uri.parse('$baseUrl/posts/$postId/likes'),
    headers: {
      'Authorization': 'Bearer $idToken',
      'X-User-Id': userId,
    },
  );
  
  if (response.statusCode == 201) {
    // Actualizar UI: incrementar contador, cambiar color de corazón
    setState(() {
      post.likesCount++;
      post.isLikedByMe = true;
    });
  }
}

// 2. Quitar like
Future<void> unlikePost(String postId) async {
  final response = await http.delete(
    Uri.parse('$baseUrl/posts/$postId/likes'),
    headers: {
      'Authorization': 'Bearer $idToken',
      'X-User-Id': userId,
    },
  );
  
  if (response.statusCode == 204) {
    setState(() {
      post.likesCount--;
      post.isLikedByMe = false;
    });
  }
}

// 3. Agregar comentario
Future<void> addComment(String postId, String text) async {
  final response = await http.post(
    Uri.parse('$baseUrl/posts/$postId/comments'),
    headers: {
      'Authorization': 'Bearer $idToken',
      'X-User-Id': userId,
      'Content-Type': 'application/json',
    },
    body: jsonEncode({'text': text}),
  );
  
  if (response.statusCode == 201) {
    final comment = Comment.fromJson(jsonDecode(response.body));
    setState(() {
      post.commentsCount++;
      comments.insert(0, comment);
    });
  }
}
```

### Flujo 5: Cargar Feed con Paginación

```dart
class FeedProvider extends ChangeNotifier {
  List<Post> posts = [];
  int currentPage = 0;
  bool isLoading = false;
  bool hasMore = true;
  
  Future<void> loadFeed({bool refresh = false}) async {
    if (isLoading) return;
    
    if (refresh) {
      currentPage = 0;
      posts.clear();
      hasMore = true;
    }
    
    isLoading = true;
    notifyListeners();
    
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/feed?limit=20&page=$currentPage'),
        headers: {
          'Authorization': 'Bearer $idToken',
          'X-User-Id': userId,
        },
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final newPosts = (data['posts'] as List)
            .map((json) => Post.fromJson(json))
            .toList();
        
        if (refresh) {
          posts = newPosts;
        } else {
          posts.addAll(newPosts);
        }
        
        hasMore = newPosts.length == 20;
        currentPage++;
      }
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }
}

// En la UI (ListView con infinite scroll):
ListView.builder(
  controller: scrollController,
  itemCount: posts.length + 1,
  itemBuilder: (context, index) {
    if (index == posts.length) {
      if (hasMore) {
        feedProvider.loadFeed();
        return Center(child: CircularProgressIndicator());
      }
      return SizedBox.shrink();
    }
    return PostCard(post: posts[index]);
  },
);
```

---

## 🚀 Iniciar los Servicios (Backend)

### Requisitos Previos
- Java 17+
- Python 3.8+
- GPU NVIDIA con CUDA (para filtros)

### 1. Iniciar API Gateway
```powershell
cd backend-java/api-gateway
.\start-gateway.ps1
```
**Puerto:** 8080  
**Verifica:** `http://localhost:8080/actuator/health`

### 2. Iniciar Auth Service
```powershell
cd backend-java/auth-service
.\start-auth.ps1
```
**Puerto:** 8082 (interno)  
**Acceso:** A través del gateway en `/api/auth/*`

### 3. Iniciar Post Service
```powershell
cd backend-java/post-service
.\start-post.ps1
```
**Puerto:** 8081 (interno)  
**Acceso:** A través del gateway en `/api/posts/*`, `/api/images/*`, `/api/feed`

### 4. Iniciar PyCUDA Service
```powershell
cd backend-java/cuda-lab-back
python -m uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```
**Puerto:** 5000 (interno)  
**Acceso:** Post-service se comunica internamente con PyCUDA
**Verifica:** `http://localhost:5000/health`

### Orden de Inicio Recomendado
1. **PyCUDA** (si vas a usar filtros)
2. **Auth Service**
3. **Post Service**
4. **API Gateway** (último, porque necesita que los otros estén arriba)

---

## 🧪 Testing

### Probar Auth Service
```powershell
cd backend-java/auth-service/docs
.\test-auth-flow.ps1
```

### Probar Post Service (sin filtros)
```powershell
cd backend-java/post-service/docs
.\test-post-flow.ps1
```

### Probar Flujo con Filtros CUDA
```powershell
cd backend-java/post-service/docs
.\test-filter-flow.ps1
```

---

## 📝 Notas Importantes para Frontend

### 1. Manejo de Tokens
- El `idToken` expira en 1 hora (3600 segundos)
- Implementar refresh automático con `refreshToken`
- Guardar tokens de forma segura (FlutterSecureStorage)
- Interceptor para agregar Authorization header automáticamente

### 2. Manejo de Imágenes
- Tamaño máximo: 10MB
- Formatos soportados: JPEG, PNG
- Comprimir imágenes antes de subir para mejor UX
- Mostrar loading indicator durante procesamiento con filtros CUDA (puede tardar 2-5 segundos)

### 3. Estados de Carga
- Implementar estados: loading, success, error
- Mostrar skeleton loaders en el feed
- Refresh to load y pull to refresh
- Manejo de errores con mensajes user-friendly

### 4. Optimizaciones
- Cachear imágenes con `cached_network_image`
- Lazy loading en el feed (paginación)
- Optimistic updates para likes (actualizar UI inmediatamente, rollback si falla)
- Debouncing en búsquedas

### 5. Offline First (Opcional)
- Guardar feed en local storage
- Sincronizar cuando vuelva conexión
- Mostrar mensaje "Sin conexión"

---

## 📞 Contacto y Soporte

**Proyecto:** UPSGlam 2.0  
**Universidad:** Universidad Politécnica Salesiana  
**Deadline:** Diciembre 10, 2025

**Equipo Backend:**
- Auth Service: Anthony
- Post Service: Anthony  
- PyCUDA Service: Equipo CUDA

**Documentación Adicional:**
- Gateway: `backend-java/api-gateway/README.md`
- Auth: `backend-java/auth-service/docs/`
- Post: `backend-java/post-service/docs/`
- CUDA: `backend-java/cuda-lab-back/QUICKSTART.md`

---

## 🎯 Quick Start para Testing

### 1. Usuario de Prueba
Si necesitas un usuario pre-creado para testing:

```
Email: testpost@ups.edu.ec
Password: test123456
Username: testpost
```

### 2. Verificar que Todo Funciona

```dart
// Test rápido
Future<bool> checkBackendHealth() async {
  try {
    // 1. Gateway
    final gatewayResponse = await http.get(
      Uri.parse('http://localhost:8080/actuator/health')
    );
    
    // 2. PyCUDA
    final cudaResponse = await http.get(
      Uri.parse('http://localhost:5000/health')
    );
    
    return gatewayResponse.statusCode == 200 && 
           cudaResponse.statusCode == 200;
  } catch (e) {
    return false;
  }
}
```

---

## ✅ Checklist de Implementación

### Autenticación
- [ ] Pantalla de registro
- [ ] Pantalla de login
- [ ] Almacenamiento seguro de tokens
- [ ] Refresh automático de tokens
- [ ] Logout
- [ ] Perfil de usuario
- [ ] Editar perfil

### Posts
- [ ] Feed principal con scroll infinito
- [ ] Crear post (subir imagen)
- [ ] Selector de filtros CUDA
- [ ] Previsualización con filtro
- [ ] Detalle de post
- [ ] Perfil con posts del usuario
- [ ] Editar caption
- [ ] Eliminar post

### Interacciones
- [ ] Dar like (doble tap + botón)
- [ ] Quitar like
- [ ] Lista de likes
- [ ] Agregar comentario
- [ ] Lista de comentarios
- [ ] Eliminar comentario (propio)

### UX/UI
- [ ] Loading states
- [ ] Error handling
- [ ] Skeleton loaders
- [ ] Pull to refresh
- [ ] Image caching
- [ ] Optimistic updates

---

**¡Éxito con la implementación! 🚀📱**
