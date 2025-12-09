🎯 0. Arquitectura general del frontend
●	Capas lógicas:

○	api / services: funciones que llaman a los endpoints (/auth/..., /feed, etc.).

○	stores / state / blocs: manejan estado global (usuario logueado, feed, etc.).

○	ui / screens: pantallas y componentes.

●	Estado global mínimo:

○	currentUser: User | null

○	authToken: string | null (idToken)

○	feed: List<Post>

○	selectedPost: Post?

●	URL base de la API (configurable):

○	Ej: BASE_URL = http://<ip-local>:8080

________________________________________
1. Módulo AUTH (login / registro / sesión)
1.1. Pantallas
●	Splash Screen

○	Revisa si hay authToken guardado (SecureStorage/SharedPrefs).

○	Si hay token:

■	Llama GET /auth/me.

■	Si responde 200 → va directo a Home.

■	Si 401 → borra token y va a Login.

●	Login Screen

○	Inputs:

■	identifier (email/username/teléfono).

■	password.

○	Botones:

■	Login

■	Ir a Register

●	Register Screen

○	Inputs:

■	email

■	password

■	confirmPassword (validado solo en frontend).

■	fullName

■	username.

○	Botones:

■	Register

■	Ir a Login

________________________________________
1.2. Lógica con endpoints
🔹 Login – POST /auth/login
●	Cuando el usuario presiona Login:

POST /auth/login
Content-Type: application/json

{
  "identifier": "pepito",
  "password": "12345678"
}

●	Si 200 OK:

○	Guardar:

■	authToken = token.idToken

■	currentUser = user

○	Navegar a Home (Feed).

●	Si 401:

○	Mostrar mensaje: “Usuario o contraseña incorrectos”.

________________________________________
🔹 Register – POST /auth/register
●	Cuando presiona Register después de validar password/confirm:

POST /auth/register
Content-Type: application/json

{
  "email":    "user@ups.edu.ec",
  "password": "12345678",
  "fullName": "Pepito Pérez",
  "username": "pepito"
}

●	Si 200:

○	Igual que login → guardar token + user y navegar a Home.

●	Si 400 / 409:

○	Mostrar mensaje según código:

■	EMAIL_ALREADY_IN_USE

■	USERNAME_ALREADY_IN_USE

■	etc.

________________________________________
🔹 Cargar usuario al abrir la app – GET /auth/me
●	En Splash o al abrir Perfil:

GET /auth/me
Authorization: Bearer <idToken>

●	Si 200:

○	Actualizar currentUser en el state.

●	Si 401:

○	Borrar token y mandar a login.

________________________________________
2. Módulo PERFIL
2.1. Pantallas
●	Profile Screen (mi perfil)

○	Muestra:

■	photoUrl

■	username

■	fullName

■	bio

■	Stats opcionales (número de posts, etc. si luego quieres).

○	Botones:

■	Editar perfil

■	Cerrar sesión

●	Edit Profile Screen

○	Inputs:

■	username

■	fullName

■	bio

○	Botones:

■	Guardar

■	Cambiar foto (abre selector de imagen → PUT avatar).

●	Profile público (de otros usuarios)

○	Parecido a mi perfil, pero sin botones de editar, y sin “cerrar sesión”.

○	Navega desde el feed cuando tocas el username/imagen del user.

________________________________________
2.2. Lógica con endpoints
🔹 Ver mi perfil – reutiliza GET /auth/me
Cada vez que entras a la pestaña Perfil:
●	Si currentUser ya está en memoria:

○	Solo renderizar.

●	Opcional: refrescar con GET /auth/me para asegurar datos actualizados.

________________________________________
🔹 Editar datos – PATCH /users/me
●	En Edit Profile, al presionar Guardar:

PATCH /users/me
Authorization: Bearer <idToken>
Content-Type: application/json

{
  "username": "pepito_cuda",
  "fullName": "Pepito Pérez",
  "bio": "Ahora programando en PyCUDA 🤖"
}

●	Si 200:

○	Actualizar currentUser en memoria.

○	Volver a pantalla de perfil.

●	Si 400 (USERNAME_ALREADY_IN_USE):

○	Mostrar error en el field username.

________________________________________
🔹 Cambiar foto de perfil – PUT /users/me/avatar
●	Flujo:

○	Usuario toca Cambiar foto.

○	Front abre galería / cámara.

○	Recibes un imageFile.

○	Haces multipart/form-data:

PUT /users/me/avatar
Authorization: Bearer <idToken>
Content-Type: multipart/form-data

image: archivo (foto)

●	Si 200:

○	Actualizar currentUser.photoUrl.

○	Refrescar imagen en toda la app.

________________________________________
🔹 Ver perfil público – GET /users/{userId}
●	Desde el feed, al tocar un username de un post:

GET /users/uid123
Authorization: Bearer <idToken>   // o sin token si permites público

●	Mostrar pantalla con:

{
  "id": "uid123",
  "username": "pepito_cuda",
  "fullName": "Pepito Pérez",
  "photoUrl": "https://...",
  "bio": "..."
}

________________________________________
3. Módulo FEED + POSTS
3.1. Pantallas
●	Home / Feed Screen

○	Lista tipo Instagram:

■	Avatar + username.

■	Imagen (imageUrl).

■	Descripción.

■	Likes & comments count.

■	Iconos:

■	❤️ like/unlike.

■	💬 comentarios (abre panel inferior).

○	Scroll infinito:

■	Cuando llega cerca del final → GET /feed?page=....

●	Post Detail Screen (opcional)

○	Si quieres pantalla completa de un post.

○	Usa GET /posts/{postId} si necesitas data extra.

________________________________________
3.2. Lógica con endpoints
🔹 Cargar feed – GET /feed
●	Al entrar a Home por primera vez:

GET /feed?page=0&size=10
Authorization: Bearer <idToken>

●	Guardas:

○	feed.items como lista de posts.

●	Para hacer scroll infinito:

○	Llevar currentPage, totalItems.

○	Cuando scrollEnd → page++ → vuelves a llamar GET /feed.

________________________________________
🔹 Like / Unlike – POST/DELETE /posts/{postId}/likes
●	Cuando el usuario toca el corazón:

○	Si likedByMe == false:

■	POST /posts/{id}/likes.

○	Si likedByMe == true:

■	DELETE /posts/{id}/likes.

●	En ambos casos, esperas respuesta:

{
  "postId": "postId123",
  "likesCount": 6,
  "likedByMe": true
}

●	Actualizas solo ese post en el estado:

○	Cambias likesCount y likedByMe.

Para que la app se sienta rápida, puedes hacer optimistic update:
1.	Cambiar el corazón en UI primero.

2.	Mandar la petición.

3.	Si falla, revertir.

________________________________________
🔹 Ver comentarios (panel inferior) – GET /posts/{postId}/comments
●	En el feed:

○	Tocas icono 💬 → abres un bottom sheet:

■	Arriba: post resumen (username + mini imagen + descripción).

■	Abajo: lista de comentarios + input.

●	Al abrir el panel:

GET /posts/{postId}/comments?page=0&size=20
Authorization: Bearer <idToken>

●	Renderizas items[] en una lista.

●	Para cargar más (scroll dentro del panel) → page++.

________________________________________
🔹 Crear comentario – POST /posts/{postId}/comments
●	El usuario escribe y presiona “Enviar”:

POST /posts/{postId}/comments
Authorization: Bearer <idToken>
Content-Type: application/json

{
  "text": "Qué buena foto 😍"
}

●	Si 201:

○	Insertas el comentario devuelto al final de la lista.

○	Actualizas commentsCount del post en el feed (+1).

________________________________________
4. Módulo CREAR POST (cámara + filtros + CUDA preview)
4.1. Pantallas
●	Create Post Flow (tipo Instagram)

1.	Camera Screen

○	Muestra:

■	Vista de cámara.

■	Carrusel de filtros abajo (solo nombres).

○	El usuario:

■	Cambia filtro seleccionado (UI local).

■	Toma la foto y confirma.

2.	Preview Screen (con filtro aplicado)

○	Muestra:

■	Imagen filtrada que devuelve backend (imageUrl).

■	Input de descripción.

■	Botón “Publicar”.

○	Aquí usas:

■	Estrategia preview + tempImageId.

________________________________________
4.2. Preview con CUDA – POST /images/preview
●	Cuando el usuario toma la foto en la cámara con un filtro elegido:

○	Front obtiene rawImageFile (jpeg/png).

○	Llama:

POST /images/preview
Authorization: Bearer <idToken>
Content-Type: multipart/form-data

image: archivo (foto original)
filter: "ups_logo"

●	Backend responde:

{
  "tempImageId": "tmp-12345",
  "imageUrl": "https://.../temp/tmp-12345.jpg"
}

●	La app:

○	Navega a Preview Screen.

○	Muestra la imagen usando imageUrl.

○	Guarda en estado:

■	currentTempImageId = "tmp-12345".

■	currentFilter = "ups_logo".

________________________________________
4.3. Crear post definitivo – POST /posts (JSON con tempImageId)
●	En la pantalla de Preview, al tocar Publicar:

POST /posts
Authorization: Bearer <idToken>
Content-Type: application/json

{
  "tempImageId": "tmp-12345",
  "description": "Foto en la UPS 😎",
  "filter": "ups_logo"
}

●	Backend responde con el Post completo:

{
  "id": "postId123",
  "userId": "uid123",
  "username": "pepito",
  "userPhotoUrl": null,
  "imageUrl": "https://.../posts/postId123.jpg",
  "filter": "ups_logo",
  "description": "Foto en la UPS 😎",
  "createdAt": "2025-12-03T10:00:00Z",
  "likesCount": 0,
  "commentsCount": 0,
  "likedByMe": true
}

●	La app:

○	Puede:

■	Insertar este post en la lista de feed en memoria en la parte superior, o

■	Volver al feed y hacer GET /feed de nuevo.

________________________________________
5. Manejo de sesión / token en el frontend
●	Cuando obtienes token.idToken:

○	Guardarlo en storage seguro (SecureStorage / SharedPreferences).

●	Todas las peticiones al backend:

○	Añadir header:

Authorization: Bearer <idToken>

●	Si recibes 401 UNAUTHORIZED:

○	Redireccionar a Login.

○	Borrar token de disco.


2️⃣ Flujo final del usuario (de punta a punta)
Imagina el camino típico del usuario:
1. Abrir la app – Splash Screen
1.	La app arranca en Splash.

2.	Revisa si hay un idToken guardado en el dispositivo.

3.	Si hay token:

○	Llama GET /auth/me.

○	Si responde 200 → guarda el usuario en memoria → navega a Home / Feed.

○	Si responde 401 → borra token → navega a Login.

4.	Si no hay token → directo a Login.

________________________________________
2. Login / Registro
Login
1.	Usuario ve pantalla Login con:

○	identifier (email / username / teléfono).

○	password.

Pulsa Login → front llama:

 POST /auth/login
2.	
3.	Si 200:

○	Guarda token.idToken en storage.

○	Guarda user (id, username, photoUrl…) en memoria.

○	Navega a Home / Feed.

4.	Si 401:

○	Muestra “Usuario o contraseña incorrectos”.

Registro
1.	Desde Login puede tocar “Crear cuenta”.

2.	En pantalla Register llena:

○	email, password, confirmPassword, fullName, username.

3.	El front valida que password == confirmPassword.

Llama:

 POST /auth/register
4.	
5.	Si 200:

○	Igual que login → guarda token y user.

○	Navega a Home / Feed.

6.	Si 400 / 409:

○	Muestra errores de EMAIL_ALREADY_IN_USE o USERNAME_ALREADY_IN_USE.

________________________________________
3. Home / Feed (timeline principal)
1.	Al entrar en Home:

Llama:

 GET /feed?page=0&size=10
○	
○	Renderiza la lista de posts, cada card con:

■	Avatar + username (tap para ir a perfil público).

■	Imagen (imageUrl).

■	Descripción.

■	Likes & comments count.

■	Icono ❤️ (estado según likedByMe).

■	Icono 💬 para comentarios.

2.	Cuando el usuario hace scroll:

○	Si llega al final de la lista:

■	Llama GET /feed?page=1, luego page=2, etc.

○	Añade los nuevos posts a la lista.

3.	Al tocar el username:

○	Navega a Perfil público.

○	Usa GET /users/{userId}.

________________________________________
4. Likes y comentarios desde el feed
Like / Unlike
●	El usuario toca el corazón:

○	Si likedByMe == false:

■	Front puede inmediatamente cambiar a rojo (optimistic).

■	Llama POST /posts/{postId}/likes.

○	Si likedByMe == true:

■	Llama DELETE /posts/{postId}/likes.

●	Actualiza likesCount y likedByMe en el post.

Comentarios
1.	El usuario toca el icono 💬.

2.	Se abre un bottom sheet / panel inferior:

○	Arriba: info resumida del post.

○	Abajo: lista de comentarios + input de texto.

La app llama:

 GET /posts/{postId}/comments?page=0&size=20
3.	
4.	Renderiza comentarios. Si hace scroll en el panel:

○	page=1, page=2, etc. si quieres paginación.

5.	Cuando escribe un comentario y toca “Enviar”:

Llama:

 POST /posts/{postId}/comments
○	
○	Si 201:

■	Agrega el comentario a la lista.

■	Sube en memoria commentsCount del post.

________________________________________
5. Crear post (cámara + filtros PyCUDA)
5.1. Abrir cámara
1.	En Home, el usuario toca el botón “+”.

2.	Navega a Camera Screen:

○	Vista de cámara.

○	Carrusel de filtros (solo UI por ahora: ups_logo, gaussian, etc.).

3.	El usuario selecciona un filtro y toma foto.

5.2. Enviar a preview CUDA – POST /images/preview
4.	Al confirmar la foto:

Front envía:

 POST /images/preview
Authorization: Bearer <idToken>
Content-Type: multipart/form-data

image: archivo de la cámara
filter: "ups_logo"
○	
5.	Backend:

○	Llama PyCUDA.

○	Sube imagen filtrada a Storage en temp/.

Devuelve:

 {
  "tempImageId": "tmp-12345",
  "imageUrl": "https://.../temp/tmp-12345.jpg"
}
○	
6.	La app:

○	Navega a Preview Screen.

○	Muestra la imagen filtrada (imageUrl).

○	Guarda tempImageId y filter en estado local.

5.3. Publicar post definitivo – POST /posts
7.	En Preview Screen:

○	El usuario escribe una descripción.

○	Toca Publicar.

Front manda:

 POST /posts
Authorization: Bearer <idToken>
Content-Type: application/json

{
  "tempImageId": "tmp-12345",
  "description": "Foto en la UPS 😎",
  "filter": "ups_logo"
}
8.	
9.	Backend:

○	Mueve imagen de temp/ → posts/.

○	Crea documento en Firestore.

○	Devuelve el Post completo.

10.	Front:

○	Inserta ese post al inicio del feed en memoria o

○	Navega al feed y hace otro GET /feed.

________________________________________
6. Perfil y edición
Ver mi perfil
1.	Usuario abre pestaña Perfil.

2.	La app usa currentUser de memoria.

3.	Opcionalmente refresca con GET /auth/me.

Editar perfil (texto)
1.	Tocando “Editar Perfil” → EditProfile Screen.

2.	Cambia username, fullName, bio.

3.	Toca Guardar:

○	PATCH /users/me.

4.	Si 200:

○	Actualiza currentUser.

○	Vuelve a pantalla de Perfil.

Cambiar avatar
1.	En Edit Profile:

○	Toca “Cambiar foto”.

2.	Abre cámara/galería.

3.	Envía PUT /users/me/avatar con la image.

4.	Si 200:

○	Actualiza photoUrl en currentUser.

○	Refresca la UI.

Logout
1.	Botón “Cerrar sesión”.

2.	La app:

○	Borra idToken del storage.

○	Limpia currentUser y estados.

○	Navega a Login.

________________________________________
3️⃣ Arquitectura propuesta del frontend
Te propongo algo limpio y sencillo, estilo por features, que sirve igual para Flutter, Ionic o Android nativo.
3.1. Estructura de carpetas (ejemplo tipo Flutter)
lib/
├── main.dart
├── core/
│   ├── config/
│   │   └── api_config.dart         # BASE_URL, timeouts
│   ├── network/
│   │   └── api_client.dart         # cliente HTTP base
│   ├── storage/
│   │   └── auth_storage.dart       # guarda y lee idToken
│   └── models/
│       ├── user.dart               # User (igual al JSON del backend)
│       ├── post.dart               # Post
│       └── comment.dart            # Comment
├── features/
│   ├── auth/
│   │   ├── data/
│   │   │   └── auth_api.dart       # llama a /auth/register, /auth/login, /auth/me
│   │   ├── state/
│   │   │   └── auth_controller.dart# maneja currentUser + token + isLoading
│   │   └── ui/
│   │       ├── login_screen.dart
│   │       ├── register_screen.dart
│   │       └── splash_screen.dart
│   ├── profile/
│   │   ├── data/
│   │   │   └── profile_api.dart    # PATCH /users/me, PUT avatar, GET /users/{id}
│   │   ├── state/
│   │   │   └── profile_controller.dart
│   │   └── ui/
│   │       ├── profile_screen.dart
│   │       └── edit_profile_screen.dart
│   ├── feed/
│   │   ├── data/
│   │   │   └── feed_api.dart       # GET /feed
│   │   ├── state/
│   │   │   └── feed_controller.dart# lista de posts, página, loading, etc.
│   │   └── ui/
│   │       ├── feed_screen.dart
│   │       └── widgets/
│   │           └── post_card.dart
│   ├── post/
│   │   ├── data/
│   │   │   └── post_api.dart       # POST /posts, DELETE /posts/{id}, GET /posts/{id}
│   │   ├── state/
│   │   │   ├── create_post_controller.dart
│   │   │   └── post_detail_controller.dart
│   │   └── ui/
│   │       ├── camera_screen.dart
│   │       ├── preview_screen.dart
│   │       └── post_detail_screen.dart
│   ├── likes_comments/
│   │   ├── data/
│   │   │   └── interaction_api.dart# likes + comments endpoints
│   │   ├── state/
│   │   │   ├── like_controller.dart
│   │   │   └── comments_controller.dart
│   │   └── ui/
│   │       └── comments_bottom_sheet.dart
└── widgets/
    └── common_button.dart          # componentes compartidos

3.2. Capas por feature
Para cada feature (auth, profile, feed, post, likes_comments) tienes:
●	data/:

○	Llama a la API usando api_client.

○	Un archivo por grupo de endpoints: auth_api.dart, feed_api.dart, etc.

●	state/:

○	Controladores / BLoC / Riverpod providers.

○	Encapsulan:

■	loading, error, data, actions.

●	ui/:

○	Pantallas y widgets.

○	Solo consumen el state y llaman acciones (no conocen endpoints).

3.3. Principios que estarías cumpliendo
●	Separation of concerns:

○	UI no conoce URLs ni tokens → solo llama a métodos del controller.

●	Reusabilidad:

○	models/ compartidos entre features (User, Post, Comment).

●	Configuración centralizada:

○	BASE_URL en core/config/api_config.dart.

○	Manejo de token en core/storage/auth_storage.dart.

●	Testabilidad:

○	Puedes testear auth_api o feed_controller mockeando el api_client.













































Auth / Perfil
●	POST /auth/register → pantalla Register ✔

●	POST /auth/login → pantalla Login ✔

●	GET /auth/me → Splash / Perfil / refrescar user ✔

●	PATCH /users/me → Editar perfil (texto) ✔

●	PUT /users/me/avatar → Cambiar foto de perfil ✔

●	GET /users/{userId} → Ver perfil de otro usuario desde el feed ✔

Posts / Feed / Imágenes
●	GET /feed → Home / Feed con scroll infinito ✔

●	POST /images/preview → pantalla de Preview con filtro CUDA ✔

●	POST /posts (con tempImageId) → Publicar post ✔

●	POST /posts (multipart directo) → si algún día quieres modo sin preview, también ✔

●	GET /posts/{postId} → opcional para pantalla de detalle de post 🔸

●	DELETE /posts/{postId} → opcional para botón “eliminar post” del dueño 🔸

Likes
●	POST /posts/{postId}/likes → botón de like ✔

●	DELETE /posts/{postId}/likes → unlike ✔

●	GET /posts/{postId}/likes → opcional pantalla “a quién le gusta esto” 🔸

Comments
●	GET /posts/{postId}/comments → panel inferior con comentarios ✔

●	POST /posts/{postId}/comments → enviar comentario ✔

●	DELETE /posts/{postId}/comments/{commentId} → opcional borrar comentario propio 🔸

●	GET /posts/{postId}/comments/stream → solo si te quieres lucir con tiempo real PRO 🔸

Todo lo core de la app tipo Instagram está cubierto. Lo marcado con 🔸 es opcional a nivel UI.

