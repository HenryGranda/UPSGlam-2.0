2. Modelo de Datos – Supabase (Postgres + Auth + Storage) 🗃️

Supabase usa:

Postgres para datos relacionales.

Supabase Auth para credenciales/usuarios base (auth.users).

Supabase Storage para archivos (imágenes).

Lo que antes eran colecciones y subcolecciones de Firestore, ahora son tablas y foreign keys.

2.1. Resumen de tablas

En el esquema public:

profiles → perfil de usuario (equivalente a users/{uid} en Firestore).

posts → publicaciones del feed (equivalente a posts/{postId}).

post_likes → quién dio like a cada post (equivalente a posts/{postId}/likes/{userId}).

comments → comentarios de cada post (equivalente a posts/{postId}/comments/{commentId}).

Notas importantes (igual que antes):

likedByMe NO se guarda en la BD, se calcula en el backend:
“¿existe un registro en post_likes para este post_id y este user_id?”.

Tokens y passwords van en Supabase Auth, no en tus tablas.

2.2. Tabla profiles (antes colección users)

En Supabase ya existe auth.users (donde viven email, password hash, etc.).
Normalmente creas una tabla profiles que referencia auth.users.id.

Tabla: public.profiles

Ejemplo de fila:

{
  "id": "uuid-igual-a-auth-users",
  "email": "user@ups.edu.ec",
  "username": "pepito",
  "full_name": "Pepito Pérez",
  "avatar_url": "https://.../avatars/uid123.jpg",
  "bio": "Estudiante de la UPS y fan de CUDA 😎",
  "created_at": "2025-12-03T10:00:00Z",
  "updated_at": "2025-12-03T10:10:00Z"
}


Campos sugeridos:

id (uuid, PK, FK → auth.users.id)

email (text) – copiado de auth.users.email para consulta rápida.

username (text) – UNIQUE, se usa en posts y comentarios.

full_name (text)

avatar_url (text, nullable) – URL en Supabase Storage (avatars/{id}.jpg).

bio (text, nullable)

created_at (timestamp with time zone, default now())

updated_at (timestamp with time zone)

El backend puede mapear a JSON camelCase:

{
  "id": "...",
  "email": "...",
  "username": "...",
  "fullName": "...",
  "photoUrl": "...",
  "bio": "..."
}


Endpoints que usan esta tabla:

GET /auth/me

PATCH /users/me

PUT /users/me/avatar

GET /users/{userId}

Indirectamente: GET /feed, GET /posts/{postId}, GET /posts/{postId}/comments (cuando quieras mostrar username/avatar actualizados).

2.3. Tabla posts (antes colección posts)

Tabla: public.posts

Ejemplo de fila:

{
  "id": "postId123",
  "user_id": "uid123",
  "username": "pepito",
  "user_photo_url": "https://.../avatars/uid123.jpg",
  "image_url": "https://.../posts/postId123.jpg",
  "filter": "ups_logo",
  "description": "Foto en la UPS 😎",
  "created_at": "2025-12-03T10:00:00Z",
  "likes_count": 5,
  "comments_count": 2
}


Campos:

id (uuid / text, PK) – equivalente a postId.

user_id (uuid, FK → profiles.id) – autor.

username (text) – cache para evitar join en consultas sencillas.

user_photo_url (text, nullable) – cache del avatar del autor al momento del post.

image_url (text) – URL en Storage (posts/{postId}.jpg).

filter (text) – nombre del filtro PyCUDA (ups_logo, gaussian, etc.).

description (text, nullable).

created_at (timestamptz, default now()).

likes_count (integer, default 0) – contador denormalizado.

comments_count (integer, default 0) – contador denormalizado.

Campos que NO se guardan en la tabla pero sí en respuestas de la API:

likedByMe → calculado por backend.

id → ya es la PK, sólo se serializa a JSON.

Endpoints que usan esta tabla:

GET /feed

POST /posts (multipart o con tempImageId)

GET /posts/{postId}

DELETE /posts/{postId}

2.4. Tabla post_likes (antes subcolección likes)

En vez de subcolección, ahora es una tabla normal con clave compuesta.

Tabla: public.post_likes

Ejemplo de fila:

{
  "post_id": "postId123",
  "user_id": "uid123",
  "created_at": "2025-12-03T10:05:00Z"
}


Campos:

post_id (uuid, FK → posts.id)

user_id (uuid, FK → profiles.id)

created_at (timestamptz, default now())

Clave primaria recomendada:

PK compuesta: (post_id, user_id) → un usuario solo puede dar un like por post.

Uso en backend:

Saber si el usuario actual ya dio like:
EXISTS(SELECT 1 FROM post_likes WHERE post_id = :post AND user_id = :currentUser)

Contar likes de un post:
SELECT count(*) FROM post_likes WHERE post_id = :post
(o mantienes likes_count actualizado en posts como en Firestore).

Endpoints que usan esta tabla:

POST /posts/{postId}/likes

DELETE /posts/{postId}/likes

GET /posts/{postId}/likes (si implementas la lista de quiénes dieron like)

2.5. Tabla comments (antes subcolección comments)

Tabla: public.comments

Ejemplo de fila:

{
  "id": "commentId456",
  "post_id": "postId123",
  "user_id": "uid999",
  "username": "maria",
  "user_photo_url": "https://.../avatars/maria.jpg",
  "text": "Qué buena foto 😍",
  "created_at": "2025-12-03T10:05:00Z"
}


Campos:

id (uuid / text, PK) – commentId.

post_id (uuid, FK → posts.id)

user_id (uuid, FK → profiles.id)

username (text) – cache del nombre al momento de comentar.

user_photo_url (text, nullable) – cache del avatar.

text (text) – contenido del comentario.

created_at (timestamptz, default now())

Endpoints que usan esta tabla:

GET /posts/{postId}/comments

POST /posts/{postId}/comments

DELETE /posts/{postId}/comments/{commentId}

(Opcional) GET /posts/{postId}/comments/stream si implementas SSE.

2.6. Relaciones entre entidades (versión SQL)

En Postgres se ve así:

User → Profile

auth.users.id 1–1 profiles.id

User → Posts

profiles.id 1–N posts.user_id

Post → Likes

posts.id 1–N post_likes.post_id

Post → Comments

posts.id 1–N comments.post_id

Diagrama textual:

auth.users (Supabase Auth)
   │ 1
   └─< profiles.id

profiles.id
   └─< posts.user_id
         ├─< post_likes.post_id
         └─< comments.post_id

2.7. Supabase Storage (equivalente a Firebase Storage)

Puedes crear un bucket (por ejemplo) upsglam y seguir la misma estructura lógica:

Avatares:

avatars/{user_id}.jpg → usado en profiles.avatar_url.

Imágenes temporales (preview CUDA):

temp/{tempImageId}.jpg

Imágenes finales de posts:

posts/{postId}.jpg

Los campos de BD que guardan URLs (avatar_url, image_url) son simplemente las URLs públicas o firmadas que te da Supabase Storage.

2.8. RLS (muy resumido, por si lo incluyes)

En Supabase casi seguro vas a querer Row Level Security:

profiles:

Usuarios sólo pueden UPDATE su propio perfil (id = auth.uid()).

posts:

INSERT: user_id = auth.uid().

DELETE: sólo el dueño del post.

post_likes:

INSERT/DELETE: user_id = auth.uid().

comments:

INSERT: user_id = auth.uid().

DELETE: autor del comentario o dueño del post.