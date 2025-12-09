1. AUTH
Endpoints
●	POST /auth/register

●	POST /auth/login

●	GET /auth/me

________________________________________
1.1. POST /auth/register
Campos del template REGISTER
●	Email → email

●	Password → password

●	Confirm password → (se valida solo en frontend, al backend llega un password)

●	Full Name → fullName

●	Username → username

________________________________________
Request (body JSON)
{
  "email": "user@ups.edu.ec",
  "password": "12345678",
  "fullName": "Pepito Pérez",
  "username": "pepito"
}

Reglas mínimas sugeridas (validaciones)
●	email: obligatorio, formato email.

●	password: obligatorio, mínimo 6–8 caracteres.

●	fullName: obligatorio, 2–80 caracteres.

●	username: obligatorio, único, sin espacios, solo letras/números/guion-bajo.

________________________________________
Response (200 OK)
Al registrarse, también haces login automático devolviendo token:
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
    "idToken": "JWT_FIREBASE_O_LO_QUE_EXPONGAS",
    "refreshToken": "REFRESH_TOKEN_OPCIONAL",
    "expiresIn": 3600
  }
}

________________________________________
Errores típicos
400 Bad Request

 { "code": "VALIDATION_ERROR", "message": "Email inválido" }
●	
409 Conflict

 { "code": "EMAIL_ALREADY_IN_USE", "message": "El email ya está registrado" }
 { "code": "USERNAME_ALREADY_IN_USE", "message": "El username ya está en uso" }
●	
________________________________________
1.2. POST /auth/login
Campos en UI
●	Phone number, username, or email

●	Password

Para que sea flexible definimos:
{
  "identifier": "pepito",
  "password": "12345678"
}

identifier puede ser:
●	un email ("user@ups.edu.ec")

●	un username ("pepito")

●	(opcional) un teléfono ("+593...")

________________________________________
Request
{
  "identifier": "user@ups.edu.ec",
  "password": "12345678"
}

o
{
  "identifier": "pepito",
  "password": "12345678"
}

________________________________________
Response (200 OK)
Mismo formato que register:
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
    "idToken": "JWT_FIREBASE_O_LO_QUE_EXPONGAS",
    "refreshToken": "REFRESH_TOKEN_OPCIONAL",
    "expiresIn": 3600
  }
}

________________________________________
Errores típicos
401 Unauthorized

 { "code": "INVALID_CREDENTIALS", "message": "Usuario o contraseña incorrectos" }
●	
________________________________________
1.3. GET /auth/me
Este endpoint NO lleva body.
 El cliente solo manda el token:
GET /auth/me
Authorization: Bearer <idToken>

________________________________________
Response (200 OK)
Solo info de perfil (sin token):
{
  "id": "uid123",
  "email": "user@ups.edu.ec",
  "username": "pepito",
  "fullName": "Pepito Pérez",
  "photoUrl": "https://firebasestorage.googleapis.com/avatars/uid123.jpg",
  "bio": "Estudiante de la UPS y fan de CUDA 😎"
}

________________________________________
Errores
401 Unauthorized (token ausente o inválido)

 { "code": "UNAUTHORIZED", "message": "Token inválido o expirado" }
●	
________________________________________
2. PERFIL DE USUARIO
2.1. Modelo de usuario (User)
Este es el JSON estándar de usuario:
{
  "id": "uid123",
  "email": "user@ups.edu.ec",
  "username": "pepito",
  "fullName": "Pepito Pérez",
  "photoUrl": "https://firebasestorage.googleapis.com/avatars/uid123.jpg",
  "bio": "Estudiante de la UPS y fan de CUDA 😎"
}

En Firestore:
●	Colección: users

●	Documento: users/{uid} con esos campos.

________________________________________
2.2. PATCH /users/me – actualizar datos de perfil (texto)
Campos que puede cambiar:
●	username

●	fullName

●	bio

Request
PATCH /users/me
Authorization: Bearer <idToken>
Content-Type: application/json

Body (ejemplo):
{
  "username": "pepito_cuda",
  "fullName": "Pepito Pérez",
  "bio": "Ahora programando en PyCUDA 🤖"
}

Todos los campos son opcionales: solo se actualiza lo que venga en el JSON.
________________________________________
Response (200 OK)
{
  "id": "uid123",
  "email": "user@ups.edu.ec",
  "username": "pepito_cuda",
  "fullName": "Pepito Pérez",
  "photoUrl": "https://firebasestorage.googleapis.com/avatars/uid123.jpg",
  "bio": "Ahora programando en PyCUDA 🤖"
}

________________________________________
Errores típicos
400 Bad Request

 { "code": "USERNAME_ALREADY_IN_USE", "message": "El username ya está en uso" }
●	
●	401 Unauthorized

●	422 Unprocessable Entity (validación de longitud/formato de campos).

________________________________________
2.3. PUT /users/me/avatar – actualizar foto de perfil
Request
PUT /users/me/avatar
Authorization: Bearer <idToken>
Content-Type: multipart/form-data

Partes del form:
●	image: archivo (foto nueva de perfil).

Conceptual:
{
  // + archivo 'image'
}

________________________________________
Flujo interno (backend)
1.	Valida token → obtiene uid123.

2.	Sube la imagen a Firebase Storage:

○	avatars/uid123.jpg

3.	Genera photoUrl pública.

4.	Actualiza users/{uid123} en Firestore con ese photoUrl.

5.	(Opcional) actualiza también photoUrl en Firebase Auth.

________________________________________
Response (200 OK)
{
  "id": "uid123",
  "email": "user@ups.edu.ec",
  "username": "pepito_cuda",
  "fullName": "Pepito Pérez",
  "photoUrl": "https://firebasestorage.googleapis.com/avatars/uid123.jpg",
  "bio": "Ahora programando en PyCUDA 🤖"
}

________________________________________
2.4. GET /users/{userId} – perfil público (opcional)
Para ver el perfil del autor de un post.
Request
GET /users/uid123
Authorization: Bearer <idToken>   // o público, según decidas

Response (200 OK)
{
  "id": "uid123",
  "username": "pepito_cuda",
  "fullName": "Pepito Pérez",
  "photoUrl": "https://firebasestorage.googleapis.com/avatars/uid123.jpg",
  "bio": "Ahora programando en PyCUDA 🤖"
}

Nota: aquí puedes omitir email por privacidad.
________________________________________
3. POSTS & FEED
3.1. Modelo Post
Modelo estándar que devuelven todos los endpoints de posts/feed:
{
  "id": "postId123",
  "userId": "uid123",
  "username": "pepito",
  "userPhotoUrl": "https://firebasestorage.googleapis.com/avatars/uid123.jpg",
  "imageUrl": "https://firebasestorage.googleapis.com/ups/posts/postId123.jpg",
  "filter": "ups_logo",
  "description": "Foto en la UPS 😎",
  "createdAt": "2025-12-03T10:00:00Z",
  "likesCount": 5,
  "commentsCount": 2,
  "likedByMe": true
}

likedByMe indica si el usuario actual (según token) ya le dio like al post.
________________________________________
3.2. GET /feed – lo primero que ve la app
Request
GET /feed?page=0&size=10
Authorization: Bearer <idToken>    // podrías permitirlo sin token, pero con token puedes calcular likedByMe

Query params:
●	page: página (empieza en 0).

●	size: cuántos posts por página.

________________________________________
Response (200 OK)
{
  "page": 0,
  "size": 10,
  "totalItems": 2,
  "items": [
    {
      "id": "postId123",
      "userId": "uid123",
      "username": "pepito",
      "userPhotoUrl": null,
      "imageUrl": "https://firebasestorage.googleapis.com/ups/postId123.jpg",
      "filter": "ups_logo",
      "description": "Foto en la UPS 😎",
      "createdAt": "2025-12-03T10:00:00Z",
      "likesCount": 5,
      "commentsCount": 2,
      "likedByMe": true
    },
    {
      "id": "postId122",
      "userId": "uid999",
      "username": "maria",
      "userPhotoUrl": "https://firebasestorage.googleapis.com/avatars/maria.jpg",
      "imageUrl": "https://firebasestorage.googleapis.com/ups/postId122.jpg",
      "filter": "gaussian",
      "description": "Proyecto CUDA",
      "createdAt": "2025-12-03T09:55:00Z",
      "likesCount": 1,
      "commentsCount": 0,
      "likedByMe": false
    }
  ]
}

________________________________________
3.3. POST /posts – crear publicación (estrategia simple: subir foto directa)
Esta es la estrategia 1 (sin preview avanzada).
 Con PyCUDA se procesa en el mismo POST /posts.
Request (conceptual)
POST /posts
Authorization: Bearer <idToken>
Content-Type: multipart/form-data

Partes del form:
●	image: archivo (la foto).

●	filter: string (ej.: "ups_logo").

●	description: string (opcional).

Conceptual en JSON:
{
  "filter": "ups_logo",
  "description": "Foto en la UPS 😎"
  // + archivo 'image'
}

________________________________________
Response (201 Created)
{
  "id": "postId123",
  "userId": "uid123",
  "username": "pepito",
  "userPhotoUrl": null,
  "imageUrl": "https://firebasestorage.googleapis.com/ups/postId123.jpg",
  "filter": "ups_logo",
  "description": "Foto en la UPS 😎",
  "createdAt": "2025-12-03T10:00:00Z",
  "likesCount": 0,
  "commentsCount": 0,
  "likedByMe": true
}

________________________________________
3.4. GET /posts/{postId} – detalle de un post
Request
GET /posts/postId123
Authorization: Bearer <idToken>   // opcional

Response (200 OK)
{
  "id": "postId123",
  "userId": "uid123",
  "username": "pepito",
  "userPhotoUrl": null,
  "imageUrl": "https://firebasestorage.googleapis.com/ups/postId123.jpg",
  "filter": "ups_logo",
  "description": "Foto en la UPS 😎",
  "createdAt": "2025-12-03T10:00:00Z",
  "likesCount": 5,
  "commentsCount": 2,
  "likedByMe": true
}

________________________________________
3.5. DELETE /posts/{postId} – borrar un post
Usado cuando el dueño de la publicación la quiere eliminar.
Request
DELETE /posts/postId123
Authorization: Bearer <idToken>

Response
Opción simple:
204 No Content

O bien:
{ "message": "Post eliminado correctamente" }

________________________________________
4. IMÁGENES + PREVIEW CON PyCUDA (ESTRATEGIA 2 – AVANZADA)
4.1. POST /images/preview – aplicar filtro y obtener imagen filtrada (preview)
Se usa justo después de tomar la foto, antes de mostrar la pantalla de “crear post”.
Request
POST /images/preview
Authorization: Bearer <idToken>
Content-Type: multipart/form-data

Partes del form:
●	image: archivo (foto original tomada por la cámara).

●	filter: string con el nombre del filtro elegido:

○	"ups_logo"

○	"gaussian"

○	"sharpen"

○	"edge_detect"

Conceptual:
{
  "filter": "ups_logo"
  // + archivo 'image'
}

Flujo interno
1.	Valida el token → userId.

Llama al PyCUDA Service:

 POST http://pycuda-service:5000/filters/{filter}
Content-Type: image/jpeg
Body: [bytes de la imagen]
2.	
3.	PyCUDA devuelve la imagen filtrada (bytes).

4.	El backend sube esa imagen filtrada a Firebase Storage en una carpeta temporal:

○	temp/tmp-12345.jpg

5.	Genera tempImageId = "tmp-12345".

________________________________________
Response (200 OK)
{
  "tempImageId": "tmp-12345",
  "imageUrl": "https://firebasestorage.googleapis.com/v0/b/tu-bucket/o/temp%2Ftmp-12345.jpg?alt=media"
}

Campos:
●	tempImageId: ID que luego se usará en POST /posts.

●	imageUrl: URL de la imagen filtrada temporal, para mostrarla como preview.

________________________________________
Errores típicos
●	400 BAD_REQUEST

○	Faltan partes (image o filter), formato no válido, etc.

●	401 UNAUTHORIZED

○	Token ausente o inválido.

●	500 INTERNAL_SERVER_ERROR

○	Falla PyCUDA o Storage.

________________________________________
4.2. POST /posts – crear publicación usando tempImageId (variante con preview)
Estrategia 2: aquí POST /posts ya NO recibe archivo, solo el tempImageId.
Request
POST /posts
Authorization: Bearer <idToken>
Content-Type: application/json

Body:
{
  "tempImageId": "tmp-12345",
  "description": "Foto en la UPS 😎",
  "filter": "ups_logo"
}

Campos:
●	tempImageId: ID devuelto por /images/preview.

●	description: texto de la publicación.

●	filter: nombre del filtro aplicado (se guarda en Firestore).

________________________________________
Flujo interno
1.	Valida token → userId.

2.	Localiza la imagen temporal en Storage:

○	De: temp/tmp-12345.jpg

3.	Genera postId (id de documento en Firestore).

4.	Mueve/renombra la imagen a carpeta definitiva:

○	A: posts/postId123.jpg

Construye URL final:

 https://firebasestorage.googleapis.com/.../posts/postId123.jpg
5.	
Crea documento en posts:

 {
  "userId": "uid123",
  "username": "pepito",
  "userPhotoUrl": null,
  "imageUrl": "https://.../posts/postId123.jpg",
  "filter": "ups_logo",
  "description": "Foto en la UPS 😎",
  "createdAt": "2025-12-03T10:00:00Z",
  "likesCount": 0,
  "commentsCount": 0
}
6.	
7.	(Opcional) Borra temp/tmp-12345.jpg.

8.	Devuelve el Post creado.

________________________________________
Response (201 Created)
{
  "id": "postId123",
  "userId": "uid123",
  "username": "pepito",
  "userPhotoUrl": null,
  "imageUrl": "https://firebasestorage.googleapis.com/ups/posts/postId123.jpg",
  "filter": "ups_logo",
  "description": "Foto en la UPS 😎",
  "createdAt": "2025-12-03T10:00:00Z",
  "likesCount": 0,
  "commentsCount": 0,
  "likedByMe": true
}

________________________________________
Errores
●	400 BAD_REQUEST

○	Falta tempImageId, no existe, etc.

●	401 UNAUTHORIZED

●	404 NOT_FOUND

○	La imagen temporal tempImageId no existe (expirada o borrada).

●	500 INTERNAL_SERVER_ERROR

○	Fallo al mover imagen o guardar post.


5. LIKES 👍
5.1. Modelo de like (interno / Firestore)
En Firestore puedes guardar algo así:
●	Colección: posts/{postId}/likes/{userId}
 Documento:

{
  "userId": "uid123",
  "createdAt": "2025-12-03T10:05:00Z"
}

En las respuestas de la API no devolvemos el like individual, sino:
●	likesCount (en el Post)

●	likedByMe (en el Post y en las respuestas de like/unlike)

________________________________________
5.2. POST /posts/{postId}/likes – dar like
Se llama cuando el usuario toca el corazón vacío.
Request
POST /posts/{postId}/likes
Authorization: Bearer <idToken>

Body: vacío.
________________________________________
Flujo interno (backend)
1.	Valida token → obtiene userId.

2.	Crea (o asegura) documento en Firestore:

○	posts/{postId}/likes/{userId}.

3.	Incrementa likesCount del post (o lo recalcula).

4.	Opcionalmente guarda info para saber si el usuario ya hizo like (para cálculo de likedByMe).

5.	Devuelve estado actualizado.

________________________________________
Response (200 OK)
{
  "postId": "postId123",
  "likesCount": 6,
  "likedByMe": true
}

La app usa esto para:
●	Cambiar el icono a corazón lleno.

●	Actualizar el número de likes.

________________________________________
Errores típicos
●	401 UNAUTHORIZED
 Sin token / token inválido.

●	404 NOT_FOUND
 El post no existe.

________________________________________
5.3. DELETE /posts/{postId}/likes – quitar like
Se llama cuando el usuario toca el corazón lleno para des-likear.
Request
DELETE /posts/{postId}/likes
Authorization: Bearer <idToken>

Body: vacío.
________________________________________
Flujo interno
1.	Valida token → userId.

2.	Borra doc posts/{postId}/likes/{userId} si existe.

3.	Decrementa likesCount (o recalcula).

4.	Devuelve estado actualizado.

________________________________________
Response (200 OK)
{
  "postId": "postId123",
  "likesCount": 5,
  "likedByMe": false
}

La app:
●	Pone el corazón en gris.

●	Actualiza el número.

________________________________________
Errores típicos
●	401 UNAUTHORIZED

●	404 NOT_FOUND (post inexistente).

________________________________________
5.4. GET /posts/{postId}/likes – listar quiénes dieron like (opcional)
Solo si quieres una pantalla tipo “A X personas les gusta esto”.
Request
GET /posts/{postId}/likes?page=0&size=20
Authorization: Bearer <idToken>

________________________________________
Response (200 OK)
{
  "page": 0,
  "size": 20,
  "totalItems": 2,
  "items": [
    {
      "userId": "uid123",
      "username": "pepito",
      "userPhotoUrl": "https://firebasestorage.googleapis.com/avatars/uid123.jpg"
    },
    {
      "userId": "uid999",
      "username": "maria",
      "userPhotoUrl": "https://firebasestorage.googleapis.com/avatars/maria.jpg"
    }
  ]
}

________________________________________
6. COMMENTS 💬
6.1. Modelo Comment
Este es el formato estándar de un comentario:
{
  "id": "commentId456",
  "postId": "postId123",
  "userId": "uid999",
  "username": "maria",
  "userPhotoUrl": "https://firebasestorage.googleapis.com/avatars/maria.jpg",
  "text": "Qué buena foto 😍",
  "createdAt": "2025-12-03T10:05:00Z"
}

En Firestore:
●	posts/{postId}/comments/{commentId} con esos campos.

________________________________________
6.2. GET /posts/{postId}/comments – listar comentarios
Usado cuando en la app tocas el icono 💬 y se abre el panel desde abajo (tipo Instagram).
 El backend no sabe del “panel”, solo devuelve la lista.
Request
GET /posts/{postId}/comments?page=0&size=20
Authorization: Bearer <idToken>   // puedes permitirlo sin token si quieres

Query params:
●	page: página (0 por defecto).

●	size: cuántos comentarios.

________________________________________
Response (200 OK)
{
  "page": 0,
  "size": 20,
  "totalItems": 2,
  "items": [
    {
      "id": "commentId456",
      "postId": "postId123",
      "userId": "uid999",
      "username": "maria",
      "userPhotoUrl": "https://firebasestorage.googleapis.com/avatars/maria.jpg",
      "text": "Qué buena foto 😍",
      "createdAt": "2025-12-03T10:05:00Z"
    },
    {
      "id": "commentId457",
      "postId": "postId123",
      "userId": "uid123",
      "username": "pepito",
      "userPhotoUrl": null,
      "text": "Gracias! 😎",
      "createdAt": "2025-12-03T10:06:00Z"
    }
  ]
}

La app simplemente muestra items[] en esa mini pantalla.
________________________________________
6.3. POST /posts/{postId}/comments – crear comentario
Cuando el usuario escribe un comentario en el panel y toca “Enviar”.
Request
POST /posts/{postId}/comments
Authorization: Bearer <idToken>
Content-Type: application/json

Body:
{
  "text": "Qué buena foto 😍"
}

________________________________________
Flujo interno
1.	Valida token → userId, username, userPhotoUrl.

2.	Crea doc posts/{postId}/comments/{commentId} en Firestore.

3.	Incrementa commentsCount del post.

4.	Devuelve el comentario creado.

________________________________________
Response (201 Created)
{
  "id": "commentId456",
  "postId": "postId123",
  "userId": "uid999",
  "username": "maria",
  "userPhotoUrl": "https://firebasestorage.googleapis.com/avatars/maria.jpg",
  "text": "Qué buena foto 😍",
  "createdAt": "2025-12-03T10:05:00Z"
}

La app puede:
●	Insertar este comentario al final de la lista del panel.

●	Incrementar commentsCount del post en el feed.

________________________________________
Errores típicos
400 BAD_REQUEST
 text vacío o demasiado largo.

 { "code": "VALIDATION_ERROR", "message": "El comentario no puede estar vacío" }
●	
●	401 UNAUTHORIZED

●	404 NOT_FOUND (postId inexistente).

________________________________________
6.4. DELETE /posts/{postId}/comments/{commentId} – borrar comentario (opcional)
Solo si quieres permitir que el autor borre su propio comentario (o el dueño del post).
Request
DELETE /posts/{postId}/comments/{commentId}
Authorization: Bearer <idToken>

Response
204 No Content

(Internamente puedes decrementar commentsCount del post.)
________________________________________
6.5. (Opcional PRO) GET /posts/{postId}/comments/stream
Si quieres hacer tiempo real en serio con WebFlux, puedes exponerse como SSE:
●	GET /posts/{postId}/comments/stream

●	produces: text/event-stream

●	Respuesta: Flux<Comment>

Pero esto ya es extra para lucirte, no obligatorio.




























7. Vista general de la arquitectura
A alto nivel:
[ App móvil ]
      │  (HTTP/JSON)
      ▼
[ API Gateway (WebFlux) ]
      │
      ├──► [ Auth/User Service (WebFlux + Firebase Auth/Firestore) ]
      │
      └──► [ Post Service (WebFlux + Firestore + Storage) ]
                 │
                 └──► [ PyCUDA Service (Python + CUDA) ]

[ Firebase Auth ]      [ Firestore ]      [ Firebase Storage ]
        ▲                    ▲                    ▲
        └────────── usados desde Auth/User y Post Service ──┘

Todo lo que documentaste de API (los endpoints) lo ve la app como si fuera un solo backend, pero por dentro está distribuido.


1. API Gateway (Spring Cloud Gateway + WebFlux)
Rol principal: “puerta de entrada única”.
●	Puerto público: :8080

●	La app móvil SIEMPRE habla con http://gateway:8080.

Responsabilidades
●	Enrutar peticiones:

○	/auth/** y /users/** → auth-user-service

○	/feed, /posts/**, /images/** → post-service

●	Aplicar cross-cutting:

○	CORS, logging, rate limit (si quieres).

○	Opcional: validación de token JWT de Firebase:

■	Verifica Authorization: Bearer <idToken>.

■	Extrae uid.

■	Inyecta cabeceras como X-User-Id, X-User-Email para los servicios internos.

●	No tiene lógica de dominio (no sabe qué es un post, ni un filtro, ni likes).

Principios que cumple
●	Single Responsibility: solo gateway, no mezcla negocio.

●	Separation of Concerns: seguridad y routing desacoplados de negocio.

●	Backend for Frontend (BFF)-like: un solo endpoint para la app móvil.

________________________________________
2. Auth & User Service (WebFlux + Firebase Auth + Firestore)
Este servicio implementa todo lo relacionado con usuarios:
2.1. Endpoints públicos (tras el gateway)
●	POST /auth/register

●	POST /auth/login

●	GET /auth/me

●	PATCH /users/me

●	PUT /users/me/avatar

●	GET /users/{userId}

2.2. Responsabilidades
●	Registrar usuarios en Firebase Auth.

●	Verificar credenciales en login (Firebase Auth + Firestore para username).

●	Gestionar el perfil en colección users/{uid} de Firestore:

○	username, fullName, photoUrl, bio, etc.

●	Gestionar la imagen de avatar en Firebase Storage:

○	avatars/{uid}.jpg.

2.3. Internamente (capas sugeridas)
Arquitectura típica en capas (estilo hexagonal / clean):
●	Controller/Handler WebFlux

○	Mapea requests a handlers reactivos:

○	Devuelven Mono<UserResponse> o similar.

●	Service (dominio de auth/usuario)

○	Lógica de negocio:

■	Validar reglas de username, email, etc.

■	Coordinar Firebase Auth + Firestore.

●	Ports & Adapters (infra)

○	FirebaseAuthClient (adaptador al SDK de Firebase Auth).

○	UserRepository (Firestore).

○	StorageClient (Firebase Storage para avatar).

●	Modelos:

○	User (dominio).

○	UserDocument (Firestore).

○	UserResponse (DTO de salida).

Principios de diseño aplicados
●	SRP: este servicio solo se encarga de usuarios/autenticación.

●	Bounded Context: “User Management” está aislado.

●	Ports & Adapters: Firebase es un detalle, reemplazable.

●	Programación reactiva: usar Mono.fromFuture(...) para integrar Firebase async con WebFlux.

________________________________________
3. Post Service (WebFlux + Firestore + Storage + PyCUDA)
Este es tu servicio “gordo” de dominio de la app social: posts, feed, likes, comentarios, filtros.
3.1. Endpoints públicos (tras el gateway)
●	Feed & posts:

○	GET /feed

○	POST /posts (versión simple multipart)

○	POST /posts (versión JSON con tempImageId)

○	GET /posts/{postId}

○	DELETE /posts/{postId}

●	Imágenes / CUDA:

○	POST /images/preview

●	Likes:

○	POST /posts/{postId}/likes

○	DELETE /posts/{postId}/likes

○	GET /posts/{postId}/likes (opcional)

●	Comments:

○	GET /posts/{postId}/comments

○	POST /posts/{postId}/comments

○	DELETE /posts/{postId}/comments/{commentId} (opcional)

○	GET /posts/{postId}/comments/stream (super pro, opcional)

3.2. Responsabilidades principales
●	Gestión de posts:

○	Crear posts con imagen + filtro.

○	Obtener feed paginado (GET /feed).

○	Obtener detalle de post.

○	Eliminar post (dueño).

●	Integración con PyCUDA:

○	POST /images/preview → llama a pycuda-service.

○	Variante 1: procesa dentro de POST /posts.

○	Variante 2: preview con tempImageId y luego crear post.

●	Persistencia de posts en Firestore:

○	Colección posts/{postId}.

○	Campos: userId, imageUrl, filter, description, createdAt, likesCount, commentsCount, etc.

●	Likes:

○	Colección posts/{postId}/likes/{userId} en Firestore.

○	Mantener likesCount y likedByMe.

●	Comentarios:

○	Colección posts/{postId}/comments/{commentId}.

○	Mantener commentsCount.

●	Imágenes:

○	Subir/borrar/mover imágenes en Firebase Storage:

■	temp/{tempImageId}.jpg

■	posts/{postId}.jpg

3.3. Capas internas (diseño)
Súper recomendable dividir internamente en subdominios:
a) Módulo post (dominio de publicaciones)
●	PostController / PostHandler (WebFlux):

○	/feed

○	/posts

○	/posts/{postId}

●	PostService:

○	Crear post desde imagen directa (estrategia 1).

○	Crear post desde tempImageId (estrategia 2).

○	Listar feed.

○	Obtener detalle.

○	Borrar post.

●	PostRepository (Firestore):

○	savePost(Post)

○	findById(postId)

○	findFeed(page, size)

b) Módulo media (imágenes + CUDA)
●	ImageController / ImageHandler:

○	POST /images/preview

●	ImageService:

○	Encargado de:

■	Enviar imagen a pycuda-service.

■	Recibir imagen filtrada.

■	Guardar en Storage como temp/{tempImageId}.jpg.

■	Mover de temp/ a posts/.

○	No sabe nada de likes ni de comments → solo media.

●	PyCudaClient (adaptador HTTP reactivo usando WebClient):

○	applyFilter(filterName, imageBytes): Mono<byte[]>

●	StorageClient (Firebase Storage):

○	uploadTemp(imageBytes) -> tempImageId + imageUrl

○	moveTempToPost(tempImageId, postId) -> imageUrlFinal

c) Módulo likes
●	LikeController:

○	POST /posts/{postId}/likes

○	DELETE /posts/{postId}/likes

○	GET /posts/{postId}/likes (opcional)

●	LikeService:

○	Agregar/quitar like.

○	Calcular/actualizar likesCount.

○	Saber likedByMe.

●	LikeRepository (Firestore):

○	Colección posts/{postId}/likes.

d) Módulo comments
●	CommentController:

○	GET /posts/{postId}/comments

○	POST /posts/{postId}/comments

○	DELETE /posts/{postId}/comments/{commentId}

●	CommentService:

○	Listar comentarios paginados.

○	Crear comentario.

○	Borrar comentario.

○	Incrementar/decrementar commentsCount.

●	CommentRepository (Firestore):

○	Colección posts/{postId}/comments.

3.4. Principios de diseño aplicados
●	Single Responsibility:

○	Media no mezcla likes.

○	Comments no mezcla PyCUDA.

○	PostService orquesta pero no sube archivos directamente (usa ImageService).

●	Separation of Concerns:

○	Controladores HTTP vs servicios de dominio vs adaptadores a Firebase/PyCUDA.

●	Hexagonal / Ports & Adapters:

○	PostService + LikeService + CommentService representan el dominio.

○	Firestore, Storage, PyCUDA son adaptadores externos.

●	Programación reactiva:

○	Usar Flux/Mono en controladores:

■	Flux<PostResponse> para feed.

■	Flux<CommentResponse> para comentarios.

○	Comunicación HTTP entre servicios: WebClient (no RestTemplate).

________________________________________
4. PyCUDA Service (Python + Flask/FastAPI)
Este es tu microservicio especializado en GPU.
Endpoints internos (no públicos a la app)
●	POST /filters/{filterName}
 Content-Type: image/jpeg
 Body: bytes de imagen.
 Respuesta: bytes de imagen filtrada (image/jpeg).

Responsabilidades
●	Cargar kernels CUDA (en GPU).

●	Aplicar filtros de convolución:

○	gaussian, sharpen, edge_detect, ups_logo, etc.

●	No sabe nada de usuarios, posts, Firebase… solo imágenes.

Principios
●	SRP extremo: solo hace procesamiento de imagen.

●	High Cohesion, Low Coupling: interfaz mínima (HTTP binario).

________________________________________
5. Firebase como plataforma de datos
5.1. Firebase Auth
●	Usado por Auth/User Service:

○	Registro de credenciales.

○	Login / verificación de password.

●	Opcionalmente por el Gateway (para verificar idToken y extraer uid).

5.2. Firestore
Colecciones sugeridas:
users/{uid}

 {
  "email": "user@ups.edu.ec",
  "username": "pepito",
  "fullName": "Pepito Pérez",
  "photoUrl": "...",
  "bio": "..."
}
●	
posts/{postId}

 {
  "userId": "uid123",
  "username": "pepito",
  "userPhotoUrl": "...",
  "imageUrl": "...",
  "filter": "ups_logo",
  "description": "Foto en la UPS 😎",
  "createdAt": "...",
  "likesCount": 5,
  "commentsCount": 2
}
●	
posts/{postId}/likes/{userId}

 {
  "userId": "uid123",
  "createdAt": "..."
}
●	
posts/{postId}/comments/{commentId}

 {
  "postId": "postId123",
  "userId": "uid999",
  "username": "maria",
  "userPhotoUrl": "...",
  "text": "Qué buena foto 😍",
  "createdAt": "..."
}
●	
5.3. Firebase Storage
●	Avatares:

○	avatars/{uid}.jpg

●	Imágenes temporales:

○	temp/{tempImageId}.jpg

●	Imágenes finales de posts:

○	posts/{postId}.jpg


🗂 Estructura general del repo
upsglam-backend/
├── docker-compose.yml
├── README.md
├── api-gateway/
├── auth-user-service/
├── post-service/
└── pycuda-service/

________________________________________
1️⃣ api-gateway (Spring Cloud Gateway)
Aquí casi no hay dominio, es solo configuración + filtros.
api-gateway/
├── pom.xml
└── src/
    └── main/
        ├── java/
        │   └── ec/ups/upsglam/gateway/
        │       ├── GatewayApplication.java
        │       ├── config/
        │       │   ├── GatewayRoutesConfig.java      # si configuras rutas via Java
        │       │   └── SecurityConfig.java           # filtro de auth global (opcional)
        │       └── security/
        │           └── FirebaseAuthenticationFilter.java  # valida idToken, mete X-User-Id
        └── resources/
            ├── application.yml                       # rutas, puertos, etc.
            └── bootstrap.yml                         # si usas profiles

●	Principio: este módulo casi no tiene “lógica”, solo configuración de gateway + seguridad.

________________________________________
2️⃣ auth-user-service (usuarios + auth + perfil)
Java, WebFlux, integración con Firebase Auth + Firestore + Storage.
auth-user-service/
├── pom.xml
└── src/
    └── main/
        ├── java/
        │   └── ec/ups/upsglam/auth/
        │       ├── AuthUserServiceApplication.java
        │       ├── config/
        │       │   ├── WebFluxConfig.java           # CORS, codecs, etc.
        │       │   └── FirebaseConfig.java          # beans de FirebaseApp, Firestore, Storage
        │       ├── api/                             # capa de exposición HTTP
        │       │   ├── router/
        │       │   │   └── AuthUserRouter.java      # define rutas funcionales
        │       │   └── handler/
        │       │       ├── AuthHandler.java         # /auth/register, /auth/login
        │       │       └── UserHandler.java         # /auth/me, /users/me, /users/{id}
        │       ├── domain/                          # modelos y servicios de dominio
        │       │   ├── model/
        │       │   │   └── User.java
        │       │   ├── service/
        │       │   │   ├── AuthService.java         # lógica de registro/login
        │       │   │   └── UserProfileService.java  # lógica de perfil, bio, avatar
        │       │   └── exception/
        │       │       ├── UserNotFoundException.java
        │       │       └── GlobalErrorHandler.java  # traduce excepciones a JSON de error
        │       └── infrastructure/                  # adapters a Firebase / persistence
        │           ├── firebase/
        │           │   ├── FirebaseAuthClient.java  # wrapper reactivo sobre Firebase Auth
        │           │   ├── FirestoreUserRepository.java # CRUD users/{uid}
        │           │   └── FirebaseStorageClient.java    # avatars/{uid}.jpg
        │           └── mapper/
        │               └── UserMapper.java          # User <-> DTO (UserResponse)
        └── resources/
            └── application.yml

●	api/* → solo HTTP (request/response).

●	domain/* → reglas de negocio, modelos.

●	infrastructure/* → Firebase, DB, Storage, mapeos.

________________________________________
3️⃣ post-service (posts, feed, likes, comments, PyCUDA, media)
Este va más cargadito, pero lo organizamos por subdominios: post, media, like, comment.
post-service/
├── pom.xml
└── src/
    └── main/
        ├── java/
        │   └── ec/ups/upsglam/post/
        │       ├── PostServiceApplication.java
        │       ├── config/
        │       │   ├── WebFluxConfig.java
        │       │   ├── FirebaseConfig.java          # Firestore + Storage
        │       │   └── PyCudaClientConfig.java      # WebClient para pycuda-service
        │       ├── api/
        │       │   ├── router/
        │       │   │   ├── PostRouter.java          # /feed, /posts...
        │       │   │   ├── MediaRouter.java         # /images/preview
        │       │   │   ├── LikeRouter.java          # /posts/{id}/likes
        │       │   │   └── CommentRouter.java       # /posts/{id}/comments
        │       │   └── handler/
        │       │       ├── PostHandler.java
        │       │       ├── MediaHandler.java
        │       │       ├── LikeHandler.java
        │       │       └── CommentHandler.java
        │       ├── domain/
        │       │   ├── post/
        │       │   │   ├── model/
        │       │   │   │   └── Post.java            # modelo de dominio Post
        │       │   │   ├── service/
        │       │   │   │   └── PostService.java     # feed, crear/borrar post, detalle
        │       │   │   └── dto/
        │       │   │       ├── PostRequest.java     # body de creación (multipart/JSON)
        │       │   │       └── PostResponse.java    # modelo que ya definiste con likedByMe
        │       │   ├── media/
        │       │   │   ├── service/
        │       │   │   │   └── ImageService.java    # coordina CUDA + Storage (temp/posts)
        │       │   │   └── model/
        │       │   │       └── TempImage.java       # tempImageId, imageUrl
        │       │   ├── like/
        │       │   │   ├── model/
        │       │   │   │   └── Like.java
        │       │   │   └── service/
        │       │   │       └── LikeService.java     # add/remove, obtener likesCount, likedByMe
        │       │   ├── comment/
        │       │   │   ├── model/
        │       │   │   │   └── Comment.java
        │       │   │   ├── dto/
        │       │   │   │   ├── CommentRequest.java
        │       │   │   │   └── CommentResponse.java
        │       │   │   └── service/
        │       │   │       └── CommentService.java
        │       │   └── exception/
        │       │       ├── PostNotFoundException.java
        │       │       ├── TempImageNotFoundException.java
        │       │       └── GlobalErrorHandler.java
        │       └── infrastructure/
        │           ├── firestore/
        │           │   ├── PostRepository.java          # Firestore para posts
        │           │   ├── LikeRepository.java          # posts/{id}/likes
        │           │   └── CommentRepository.java       # posts/{id}/comments
        │           ├── storage/
        │           │   └── FirebasePostStorageClient.java
        │           │       # uploadTemp(...), moveTempToPost(...)
        │           ├── pycuda/
        │           │   └── PyCudaClient.java            # WebClient -> /filters/{filter}
        │           └── mapper/
        │               ├── PostMapper.java
        │               └── CommentMapper.java
        └── resources/
            └── application.yml

Qué ganas con esta estructura
●	Sabes que TODO lo referente a post está en domain/post y api/router/handler.

●	Todo lo de CUDA + imágenes está concentrado en domain/media + infrastructure/storage + pycuda.

●	Likes y comments no contaminan el código de media.

________________________________________
4️⃣ pycuda-service (Python + CUDA)
Aquí sí usas la típica app.py o main.py, pero bien ordenadito.
pycuda-service/
├── requirements.txt
├── Dockerfile
└── src/
    ├── app.py                         # punto de entrada Flask/FastAPI
    ├── api/
    │   └── filters_controller.py      # endpoint /filters/{filterName}
    ├── core/
    │   ├── cuda_context.py            # inicialización de contexto CUDA
    │   ├── kernels/
    │   │   ├── gaussian.cu
    │   │   ├── sharpen.cu
    │   │   ├── edge_detect.cu
    │   │   └── ups_logo.cu
    │   └── filter_service.py          # aplica filtros usando PyCUDA
    └── utils/
        ├── image_io.py                # leer bytes -> np.array, array -> bytes
        └── config.py

●	api/filters_controller.py:

○	POST /filters/{filterName} recibe imagen y llama a filter_service.

●	core/filter_service.py:

○	Selecciona kernel según filterName, ejecuta en GPU y devuelve bytes.

________________________________________
5️⃣ Raíz del repo + Docker Compose
Para que toda la arquitectura viva feliz en una red Docker:
upsglam-backend/
├── docker-compose.yml
├── api-gateway/
├── auth-user-service/
├── post-service/
└── pycuda-service/

Ejemplo muy resumido de docker-compose.yml:
services:
  api-gateway:
    build: ./api-gateway
    ports:
      - "8080:8080"
    depends_on:
      - auth-user-service
      - post-service

  auth-user-service:
    build: ./auth-user-service
    depends_on:
      - firebase-emulator # si lo usas local

  post-service:
    build: ./post-service
    depends_on:
      - pycuda-service

  pycuda-service:
    build: ./pycuda-service
    # aquí montas GPU si tienes

  # opcional: firebase emulator para local
  firebase-emulator:
    image: somedev/firebase-emulator



ESQUELETO 


🧩 1. post-service – estructura de paquetes
ec.ups.upsglam.post
├── PostServiceApplication.java
├── config/
│   ├── WebFluxConfig.java
│   ├── FirebaseConfig.java
│   └── PyCudaClientConfig.java
├── api/
│   ├── router/
│   │   ├── PostRouter.java
│   │   ├── MediaRouter.java
│   │   ├── LikeRouter.java
│   │   └── CommentRouter.java
│   └── handler/
│       ├── PostHandler.java
│       ├── MediaHandler.java
│       ├── LikeHandler.java
│       └── CommentHandler.java
├── domain/
│   ├── post/
│   │   ├── model/Post.java
│   │   ├── dto/PostResponse.java
│   │   └── service/PostService.java
│   ├── media/
│   │   ├── model/TempImage.java
│   │   └── service/ImageService.java
│   ├── like/
│   │   ├── model/Like.java
│   │   └── service/LikeService.java
│   ├── comment/
│   │   ├── model/Comment.java
│   │   ├── dto/CommentRequest.java
│   │   └── service/CommentService.java
│   └── exception/...
└── infrastructure/
    ├── firestore/
    │   ├── PostRepository.java
    │   ├── LikeRepository.java
    │   └── CommentRepository.java
    ├── storage/FirebasePostStorageClient.java
    └── pycuda/PyCudaClient.java

________________________________________
1.1. Router + Handler para POSTS
PostRouter.java
package ec.ups.upsglam.post.api.router;

import ec.ups.upsglam.post.api.handler.PostHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.server.RouterFunction;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Configuration
public class PostRouter {

    @Bean
    public RouterFunction<?> postRoutes(PostHandler handler) {
        return route(GET("/feed"), handler::getFeed)
            .andRoute(POST("/posts").and(contentType(org.springframework.http.MediaType.MULTIPART_FORM_DATA)), handler::createPostMultipart)
            .andRoute(POST("/posts").and(contentType(org.springframework.http.MediaType.APPLICATION_JSON)), handler::createPostFromTemp)
            .andRoute(GET("/posts/{postId}"), handler::getPostById)
            .andRoute(DELETE("/posts/{postId}"), handler::deletePost);
    }
}

PostHandler.java (esqueleto)
package ec.ups.upsglam.post.api.handler;

import ec.ups.upsglam.post.domain.post.service.PostService;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

@Component
public class PostHandler {

    private final PostService postService;

    public PostHandler(PostService postService) {
        this.postService = postService;
    }

    public Mono<ServerResponse> getFeed(ServerRequest request) {
        int page = Integer.parseInt(request.queryParam("page").orElse("0"));
        int size = Integer.parseInt(request.queryParam("size").orElse("10"));
        String userId = request.headers().firstHeader("X-User-Id"); // opcional, desde gateway

        return postService.getFeed(page, size, userId)
                .flatMap(feedPage ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(feedPage)
                );
    }

    // Estrategia 1: multipart (image + filter)
    public Mono<ServerResponse> createPostMultipart(ServerRequest request) {
        String userId = request.headers().firstHeader("X-User-Id");
        return postService.createPostFromMultipart(request, userId)
                .flatMap(post ->
                        ServerResponse.status(201)
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(post)
                );
    }

    // Estrategia 2: JSON con tempImageId
    public Mono<ServerResponse> createPostFromTemp(ServerRequest request) {
        String userId = request.headers().firstHeader("X-User-Id");
        return request.bodyToMono(CreatePostFromTempRequest.class)
                .flatMap(body -> postService.createPostFromTemp(body, userId))
                .flatMap(post ->
                        ServerResponse.status(201)
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(post)
                );
    }

    public Mono<ServerResponse> getPostById(ServerRequest request) {
        String postId = request.pathVariable("postId");
        String userId = request.headers().firstHeader("X-User-Id");
        return postService.getPostById(postId, userId)
                .flatMap(post ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(post)
                );
    }

    public Mono<ServerResponse> deletePost(ServerRequest request) {
        String postId = request.pathVariable("postId");
        String userId = request.headers().firstHeader("X-User-Id");
        return postService.deletePost(postId, userId)
                .then(ServerResponse.noContent().build());
    }

    // DTO para estrategia 2
    public static class CreatePostFromTempRequest {
        public String tempImageId;
        public String description;
        public String filter;
    }
}

________________________________________
1.2. Servicio de dominio de POSTS
PostService.java (esqueleto)
package ec.ups.upsglam.post.domain.post.service;

import ec.ups.upsglam.post.api.handler.PostHandler.CreatePostFromTempRequest;
import ec.ups.upsglam.post.domain.media.service.ImageService;
import ec.ups.upsglam.post.domain.post.dto.PostResponse;
import ec.ups.upsglam.post.infrastructure.firestore.PostRepository;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.server.ServerRequest;
import reactor.core.publisher.Mono;

@Service
public class PostService {

    private final PostRepository postRepository;
    private final ImageService imageService;

    public PostService(PostRepository postRepository, ImageService imageService) {
        this.postRepository = postRepository;
        this.imageService = imageService;
    }

    public Mono<FeedPage<PostResponse>> getFeed(int page, int size, String currentUserId) {
        return postRepository.findFeed(page, size, currentUserId);
    }

    public Mono<PostResponse> getPostById(String postId, String currentUserId) {
        return postRepository.findById(postId, currentUserId);
    }

    // Estrategia 1: Multipart (ServerRequest porque viene multipart/form-data)
    public Mono<PostResponse> createPostFromMultipart(ServerRequest request, String userId) {
        // 1) extraer image + filter + description del multipart
        // 2) imageService.processAndUpload(image, filter) -> imageUrl
        // 3) crear documento Post en Firestore -> PostResponse
        return imageService.processAndUploadMultipart(request, userId)
                .flatMap(imageResult ->
                        postRepository.saveNewPost(
                                userId,
                                imageResult.getImageUrl(),
                                imageResult.getFilter(),
                                imageResult.getDescription()
                        )
                );
    }

    // Estrategia 2: JSON con tempImageId
    public Mono<PostResponse> createPostFromTemp(CreatePostFromTempRequest body, String userId) {
        return imageService.moveTempToPost(body.tempImageId)
                .flatMap(imageUrl ->
                        postRepository.saveNewPost(
                                userId,
                                imageUrl,
                                body.filter,
                                body.description
                        )
                );
    }

    public Mono<Void> deletePost(String postId, String userId) {
        // verificar dueño, borrar post, borrar imagen si quieres
        return postRepository.deletePost(postId, userId);
    }
}

FeedPage<PostResponse> sería un DTO con page, size, totalItems, items.
________________________________________
1.3. Servicio de imágenes + PyCUDA
ImageService.java (esqueleto)
package ec.ups.upsglam.post.domain.media.service;

import ec.ups.upsglam.post.domain.media.model.TempImage;
import ec.ups.upsglam.post.infrastructure.pycuda.PyCudaClient;
import ec.ups.upsglam.post.infrastructure.storage.FirebasePostStorageClient;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.server.ServerRequest;
import reactor.core.publisher.Mono;

@Service
public class ImageService {

    private final PyCudaClient pyCudaClient;
    private final FirebasePostStorageClient storageClient;

    public ImageService(PyCudaClient pyCudaClient,
                        FirebasePostStorageClient storageClient) {
        this.pyCudaClient = pyCudaClient;
        this.storageClient = storageClient;
    }

    // Usado en POST /images/preview
    public Mono<TempImage> createPreview(FilePart imagePart, String filter) {
        return imagePart.content()
                .aggregate()
                .asByteArray()
                .flatMap(bytes -> pyCudaClient.applyFilter(filter, bytes))
                .flatMap(filteredBytes -> storageClient.uploadTemp(filteredBytes))
                // uploadTemp devuelve TempImage (tempImageId, imageUrl)
                ;
    }

    // Estrategia 1: procesar y subir en un solo paso al crear post (multipart)
    public Mono<ImageResult> processAndUploadMultipart(ServerRequest request, String userId) {
        // leer partes: image, filter, description
        return request.multipartData()
                .flatMap(parts -> {
                    FilePart imagePart = (FilePart) parts.toSingleValueMap().get("image");
                    String filter = parts.toSingleValueMap().get("filter").value();
                    String description = parts.toSingleValueMap().get("description") != null
                            ? parts.toSingleValueMap().get("description").value()
                            : null;

                    return imagePart.content().aggregate().asByteArray()
                            .flatMap(bytes -> pyCudaClient.applyFilter(filter, bytes))
                            .flatMap(filteredBytes -> storageClient.uploadPostImage(filteredBytes))
                            .map(imageUrl -> new ImageResult(imageUrl, filter, description));
                });
    }

    // Estrategia 2: mover de temp/ a posts/
    public Mono<String> moveTempToPost(String tempImageId) {
        return storageClient.moveTempToPost(tempImageId); // devuelve imageUrl final
    }

    public static class ImageResult {
        private final String imageUrl;
        private final String filter;
        private final String description;

        public ImageResult(String imageUrl, String filter, String description) {
            this.imageUrl = imageUrl;
            this.filter = filter;
            this.description = description;
        }

        public String getImageUrl() { return imageUrl; }
        public String getFilter() { return filter; }
        public String getDescription() { return description; }
    }
}

PyCudaClient.java (esqueleto)
package ec.ups.upsglam.post.infrastructure.pycuda;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Component
public class PyCudaClient {

    private final WebClient webClient;

    public PyCudaClient(@Qualifier("pycudaWebClient") WebClient webClient) {
        this.webClient = webClient;
    }

    public Mono<byte[]> applyFilter(String filterName, byte[] imageBytes) {
        return webClient.post()
                .uri("/filters/{filter}", filterName)
                .contentType(MediaType.IMAGE_JPEG)
                .bodyValue(imageBytes)
                .retrieve()
                .bodyToMono(byte[].class);
    }
}

FirebasePostStorageClient.java (esqueleto)
package ec.ups.upsglam.post.infrastructure.storage;

import com.google.cloud.storage.BlobId;
import com.google.cloud.storage.BlobInfo;
import com.google.cloud.storage.Storage;
import ec.ups.upsglam.post.domain.media.model.TempImage;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.UUID;

@Component
public class FirebasePostStorageClient {

    private final Storage storage;
    private final String bucketName = "tu-bucket";

    public FirebasePostStorageClient(Storage storage) {
        this.storage = storage;
    }

    public Mono<TempImage> uploadTemp(byte[] filteredBytes) {
        String tempId = "tmp-" + UUID.randomUUID();
        String objectName = "temp/" + tempId + ".jpg";

        return Mono.fromCallable(() -> {
            BlobId blobId = BlobId.of(bucketName, objectName);
            BlobInfo blobInfo = BlobInfo.newBuilder(blobId).setContentType("image/jpeg").build();
            storage.create(blobInfo, filteredBytes);
            String url = generatePublicUrl(objectName);
            return new TempImage(tempId, url);
        });
    }

    public Mono<String> uploadPostImage(byte[] filteredBytes) {
        String postId = "post-" + UUID.randomUUID();
        String objectName = "posts/" + postId + ".jpg";

        return Mono.fromCallable(() -> {
            BlobId blobId = BlobId.of(bucketName, objectName);
            BlobInfo blobInfo = BlobInfo.newBuilder(blobId).setContentType("image/jpeg").build();
            storage.create(blobInfo, filteredBytes);
            return generatePublicUrl(objectName);
        });
    }

    public Mono<String> moveTempToPost(String tempImageId) {
        return Mono.fromCallable(() -> {
            String tempObject = "temp/" + tempImageId + ".jpg";
            String postId = "post-" + UUID.randomUUID();
            String postObject = "posts/" + postId + ".jpg";
            storage.copy(Storage.CopyRequest.of(bucketName, tempObject, bucketName, postObject));
            storage.delete(BlobId.of(bucketName, tempObject));
            return generatePublicUrl(postObject);
        });
    }

    private String generatePublicUrl(String objectName) {
        // versión simple, depende de cómo tengas configurado el bucket
        return "https://firebasestorage.googleapis.com/v0/b/" +
                bucketName + "/o/" + objectName.replace("/", "%2F") + "?alt=media";
    }
}

TempImage.java
package ec.ups.upsglam.post.domain.media.model;

public class TempImage {
    private final String tempImageId;
    private final String imageUrl;

    public TempImage(String tempImageId, String imageUrl) {
        this.tempImageId = tempImageId;
        this.imageUrl = imageUrl;
    }

    public String getTempImageId() { return tempImageId; }
    public String getImageUrl() { return imageUrl; }
}

________________________________________
1.4. Repositorio de POSTS (Firestore)
PostRepository.java (esqueleto)
package ec.ups.upsglam.post.infrastructure.firestore;

import ec.ups.upsglam.post.domain.post.dto.PostResponse;
import reactor.core.publisher.Mono;

public interface PostRepository {

    Mono<FeedPage<PostResponse>> findFeed(int page, int size, String currentUserId);

    Mono<PostResponse> findById(String postId, String currentUserId);

    Mono<PostResponse> saveNewPost(String userId,
                                   String imageUrl,
                                   String filter,
                                   String description);

    Mono<Void> deletePost(String postId, String userId);
}

Implementación (Firestore usando Mono.fromFuture(...)) puede ir en algo como FirestorePostRepository.java.
________________________________________
2️⃣ auth-user-service – esqueletos básicos
AuthRouter.java
package ec.ups.upsglam.auth.api.router;

import ec.ups.upsglam.auth.api.handler.AuthHandler;
import ec.ups.upsglam.auth.api.handler.UserHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.server.RouterFunction;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Configuration
public class AuthRouter {

    @Bean
    public RouterFunction<?> authRoutes(AuthHandler authHandler,
                                        UserHandler userHandler) {
        return route(POST("/auth/register"), authHandler::register)
                .andRoute(POST("/auth/login"), authHandler::login)
                .andRoute(GET("/auth/me"), userHandler::me)
                .andRoute(PATCH("/users/me"), userHandler::updateMe)
                .andRoute(PUT("/users/me/avatar"), userHandler::updateAvatar)
                .andRoute(GET("/users/{userId}"), userHandler::getPublicProfile);
    }
}

AuthHandler.java (esqueleto)
package ec.ups.upsglam.auth.api.handler;

import ec.ups.upsglam.auth.domain.service.AuthService;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

@Component
public class AuthHandler {

    private final AuthService authService;

    public AuthHandler(AuthService authService) {
        this.authService = authService;
    }

    public Mono<ServerResponse> register(ServerRequest request) {
        return request.bodyToMono(RegisterRequest.class)
                .flatMap(authService::register)
                .flatMap(response ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(response)
                );
    }

    public Mono<ServerResponse> login(ServerRequest request) {
        return request.bodyToMono(LoginRequest.class)
                .flatMap(authService::login)
                .flatMap(response ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(response)
                );
    }

    public static class RegisterRequest {
        public String email;
        public String password;
        public String fullName;
        public String username;
    }

    public static class LoginRequest {
        public String identifier;
        public String password;
    }
}

AuthService.java (esqueleto)
package ec.ups.upsglam.auth.domain.service;

import ec.ups.upsglam.auth.api.handler.AuthHandler.LoginRequest;
import ec.ups.upsglam.auth.api.handler.AuthHandler.RegisterRequest;
import ec.ups.upsglam.auth.domain.model.User;
import ec.ups.upsglam.auth.infrastructure.firebase.FirebaseAuthClient;
import ec.ups.upsglam.auth.infrastructure.firebase.FirestoreUserRepository;
import reactor.core.publisher.Mono;
import org.springframework.stereotype.Service;

@Service
public class AuthService {

    private final FirebaseAuthClient firebaseAuthClient;
    private final FirestoreUserRepository userRepository;

    public AuthService(FirebaseAuthClient firebaseAuthClient,
                       FirestoreUserRepository userRepository) {
        this.firebaseAuthClient = firebaseAuthClient;
        this.userRepository = userRepository;
    }

    public Mono<AuthResponse> register(RegisterRequest request) {
        // validaciones básicas
        // crear usuario en Firebase Auth
        // crear documento en users/{uid}
        // devolver user + token
        return firebaseAuthClient.createUser(request.email, request.password)
                .flatMap(firebaseUser -> userRepository.createUserProfile(
                                firebaseUser.getUid(),
                                request.email,
                                request.username,
                                request.fullName
                        )
                        .flatMap(user -> firebaseAuthClient.createCustomToken(firebaseUser.getUid())
                                .map(token -> AuthResponse.from(user, token))
                        )
                );
    }

    public Mono<AuthResponse> login(LoginRequest request) {
        // buscar por email/username → email
        // verificar password en Firebase Auth
        // cargar perfil en Firestore
        // devolver user + token
        return firebaseAuthClient.login(request.identifier, request.password)
                .flatMap(firebaseUser -> userRepository.findById(firebaseUser.getUid())
                        .flatMap(user -> firebaseAuthClient.createCustomToken(firebaseUser.getUid())
                                .map(token -> AuthResponse.from(user, token))
                        )
                );
    }

    public static class AuthResponse {
        public User user;
        public Token token;

        public static AuthResponse from(User user, String idToken) {
            AuthResponse res = new AuthResponse();
            res.user = user;
            res.token = new Token(idToken, null, 3600);
            return res;
        }

        public static class Token {
            public String idToken;
            public String refreshToken;
            public int expiresIn;

            public Token(String idToken, String refreshToken, int expiresIn) {
                this.idToken = idToken;
                this.refreshToken = refreshToken;
                this.expiresIn = expiresIn;
            }
        }
    }
}

________________________________________
Con esto ya tienes:
●	Rutas concretas (RouterFunction).

●	Handlers concretos (PostHandler, AuthHandler, etc.).

●	Servicios de dominio (PostService, ImageService, AuthService).

●	Clientes de infraestructura (PyCudaClient, FirebasePostStorageClient).

Desde aquí es “solo” ir llenando la lógica interna, pero la columna vertebral de la arquitectura ya está.
Si quieres, en el próximo paso podemos elegir un flujo completo (por ejemplo POST /images/preview) y te lo escribo de punta a punta: handler → service → PyCudaClient → Storage, con más detalle en la parte que vayas a implementar primero.

