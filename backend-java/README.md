# 🚀 UPSGlam Backend - Microservices Architecture

## 📋 Descripción General del Backend

El backend de UPSGlam 2.0 está construido con una **arquitectura de microservicios** que combina tecnologías modernas de Java y Python para proporcionar una plataforma social escalable, reactiva y de alto rendimiento.

### 🎯 Características Principales

- **Arquitectura de Microservicios**: Servicios independientes y escalables
- **Programación Reactiva**: Spring WebFlux con patrones no bloqueantes
- **Procesamiento GPU**: Aceleración de filtros de imagen con CUDA
- **Cloud Native**: Firebase y Supabase para autenticación y almacenamiento
- **Containerización**: Docker Compose para orquestación de servicios
- **API Gateway**: Punto de entrada único con Spring Cloud Gateway

---

## 🏗️ Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                          │
│                   (Mobile App - Flutter)                  │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTP/REST
                     ↓
┌──────────────────────────────────────────────────────────┐
│                   API GATEWAY (8080)                      │
│              Spring Cloud Gateway 2023.0.0                │
│                                                            │
│  • Request Routing & Load Balancing                       │
│  • CORS Configuration                                     │
│  • Circuit Breaker Pattern                                │
│  • Reactive WebFlux                                       │
└────────┬─────────────────┬──────────────────┬────────────┘
         │                 │                  │
         ↓                 ↓                  ↓
┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│  AUTH SERVICE    │ │  POST SERVICE    │ │  CUDA BACKEND   │
│  Port: 8082      │ │  Port: 8081      │ │  Port: 5000     │
│                  │ │                  │ │                 │
│  Spring WebFlux  │ │  Spring WebFlux  │ │  Python FastAPI │
│  + Firebase SDK  │ │  + R2DBC         │ │  + PyCUDA       │
└────────┬─────────┘ └────────┬─────────┘ └────────┬────────┘
         │                    │                     │
         ↓                    ↓                     ↓
┌──────────────────┐ ┌──────────────────┐ ┌────────────────┐
│    FIREBASE      │ │    SUPABASE      │ │  NVIDIA GPU    │
│                  │ │                  │ │                │
│  • Firestore     │ │  • PostgreSQL    │ │  • CUDA 12.x   │
│  • Auth          │ │  • Storage       │ │  • Parallel    │
│  • Storage       │ │  • R2DBC Driver  │ │    Processing  │
└──────────────────┘ └──────────────────┘ └────────────────┘
```

---

## 📦 Componentes del Backend

### 1. API Gateway (Spring Cloud Gateway)
**Puerto**: 8080  
**Framework**: Spring Cloud Gateway 2023.0.0

**Responsabilidades**:
- ✅ Punto de entrada único para todas las requests
- ✅ Enrutamiento inteligente a microservicios
- ✅ Configuración CORS centralizada
- ✅ Load balancing y circuit breaker
- ✅ Implementación completamente reactiva

**Rutas Configuradas**:
```yaml
/api/auth/**    → Auth Service (8082)
/api/posts/**   → Post Service (8081)
/api/feed/**    → Post Service (8081)
/api/filters/** → CUDA Backend (5000)
```

### 2. Auth Service (Spring WebFlux + Firebase)
**Puerto**: 8082  
**Framework**: Spring Boot 3.2.0 + Spring WebFlux  
**Database**: Firebase Firestore (NoSQL)  
**Storage**: Firebase Storage

**Responsabilidades**:
- ✅ Autenticación de usuarios con Firebase Auth
- ✅ Registro y gestión de perfiles
- ✅ Sistema de seguimiento (follow/unfollow)
- ✅ Upload de avatares a Firebase Storage
- ✅ Consultas reactivas a Firestore

**Tecnologías Clave**:
- Firebase Admin SDK 9.2.0
- Spring WebFlux (Reactor Netty)
- Google Cloud Firestore
- Reactive Streams API

### 3. Post Service (Spring Boot + Supabase)
**Puerto**: 8081  
**Framework**: Spring Boot 3.2.0 + R2DBC  
**Database**: Supabase PostgreSQL  
**Storage**: Supabase Object Storage

**Responsabilidades**:
- ✅ CRUD de publicaciones
- ✅ Sistema de likes (reactivo)
- ✅ Sistema de comentarios
- ✅ Feed personalizado
- ✅ Proxy a CUDA Backend para filtros
- ✅ Upload de imágenes a Supabase

**Tecnologías Clave**:
- Spring Data R2DBC (Reactive)
- Supabase Client
- PostgreSQL 15+
- WebClient para comunicación con CUDA

### 4. CUDA Backend (Python + PyCUDA)
**Puerto**: 5000  
**Framework**: FastAPI 0.122.0  
**Compute**: NVIDIA GPU + CUDA 12.x

**Responsabilidades**:
- ✅ Procesamiento de imágenes en GPU
- ✅ 7 filtros de convolución CUDA
- ✅ Kernels optimizados en PyCUDA
- ✅ API REST para aplicar filtros

**Filtros Disponibles**:
1. Gaussian Blur
2. Box Blur
3. Prewitt Edge Detection
4. Laplacian Edge Detection
5. UPS Logo Overlay
6. Boomerang Effect
7. CR7 Mask

---

## ⚡ API Reactiva con WebFlux + Firebase

### 🔥 ¿Por qué Programación Reactiva?

UPSGlam utiliza **Spring WebFlux** con **Firebase** para implementar un backend completamente **no bloqueante** y **asíncrono**. Esto permite:

- **Alta Concurrencia**: Manejo de miles de requests simultáneas con pocos threads
- **Backpressure**: Control de flujo de datos para evitar sobrecarga
- **Escalabilidad**: Uso eficiente de recursos del servidor
- **Latencia Baja**: Operaciones de I/O no bloqueantes
- **Tiempo Real**: Actualizaciones reactivas en Firebase Firestore

### 🛠️ Stack Reactivo

```java
Spring WebFlux (Reactor)
    ↓
Reactor Netty (HTTP Server)
    ↓
Project Reactor (Mono/Flux)
    ↓
Firebase Admin SDK (Async API)
    ↓
Google Cloud Firestore
```

### 📚 Conceptos Clave de Programación Reactiva

#### 1. **Mono y Flux**

```java
// Mono: 0 o 1 elemento (operaciones individuales)
Mono<User> user = userService.getUserById(userId);

// Flux: 0 a N elementos (streams de datos)
Flux<Post> posts = postService.getAllPosts();
```

#### 2. **Operadores Reactivos**

```java
// map: Transformar datos
Mono<UserDTO> userDto = userMono.map(user -> new UserDTO(user));

// flatMap: Operaciones asíncronas encadenadas
Mono<Post> postWithUser = postService.getPost(postId)
    .flatMap(post -> userService.getUser(post.getUserId())
        .map(user -> {
            post.setUser(user);
            return post;
        }));

// filter: Filtrar elementos
Flux<Post> activePosts = allPosts.filter(post -> post.isActive());

// switchIfEmpty: Valor por defecto
Mono<User> user = userService.findByEmail(email)
    .switchIfEmpty(Mono.error(new UserNotFoundException()));
```

#### 3. **Composición Asíncrona**

```java
// Ejecutar múltiples operaciones en paralelo
Mono<PostResponse> response = Mono.zip(
    postService.getPost(postId),
    likeService.countLikes(postId),
    commentService.countComments(postId)
).map(tuple -> new PostResponse(
    tuple.getT1(),  // Post
    tuple.getT2(),  // Like count
    tuple.getT3()   // Comment count
));
```

### 🔥 Integración WebFlux + Firebase

#### **Problema**: Firebase Admin SDK no es nativo reactivo

La Firebase Admin SDK usa callbacks y `ApiFuture<T>`, no `Mono<T>` ni `Flux<T>`.

#### **Solución**: Adaptar Firebase a Reactor

```java
// Convertir ApiFuture<T> a Mono<T>
public Mono<User> getUserFromFirestore(String userId) {
    return Mono.fromFuture(() -> {
        ApiFuture<DocumentSnapshot> future = firestore
            .collection("users")
            .document(userId)
            .get();
        
        return future.toCompletableFuture();
    })
    .map(snapshot -> snapshot.toObject(User.class))
    .switchIfEmpty(Mono.error(new UserNotFoundException()));
}
```

#### **Ejemplo Real: Auth Service Login**

```java
@PostMapping("/login")
public Mono<LoginResponse> login(@RequestBody LoginRequest request) {
    return Mono.fromCallable(() -> 
        // 1. Verificar credenciales con Firebase Auth (I/O no bloqueante)
        FirebaseAuth.getInstance()
            .getUserByEmail(request.getEmail())
    )
    .flatMap(userRecord -> 
        // 2. Buscar datos adicionales en Firestore (async)
        getUserFromFirestore(userRecord.getUid())
    )
    .flatMap(user -> 
        // 3. Validar contraseña (puede ser async)
        validatePassword(user, request.getPassword())
    )
    .map(user -> 
        // 4. Generar token JWT
        new LoginResponse(user, generateToken(user))
    )
    .onErrorMap(e -> new AuthenticationException("Login failed", e));
    // Todo esto sin bloquear threads!
}
```

### 🔄 Operaciones CRUD Reactivas en Firestore

#### **Create (POST)**

```java
@PostMapping("/users")
public Mono<User> createUser(@RequestBody User user) {
    return Mono.fromFuture(() -> {
        DocumentReference docRef = firestore
            .collection("users")
            .document(user.getId());
        
        return docRef.set(user).toCompletableFuture();
    })
    .thenReturn(user)
    .doOnSuccess(u -> log.info("User created: {}", u.getId()));
}
```

#### **Read (GET)**

```java
@GetMapping("/users/{id}")
public Mono<User> getUser(@PathVariable String id) {
    return Mono.fromFuture(() -> 
        firestore.collection("users")
            .document(id)
            .get()
            .toCompletableFuture()
    )
    .map(snapshot -> snapshot.toObject(User.class))
    .switchIfEmpty(Mono.error(new NotFoundException("User not found")));
}
```

#### **Update (PUT)**

```java
@PutMapping("/users/{id}")
public Mono<User> updateUser(@PathVariable String id, @RequestBody User user) {
    return Mono.fromFuture(() -> 
        firestore.collection("users")
            .document(id)
            .set(user, SetOptions.merge())
            .toCompletableFuture()
    )
    .thenReturn(user);
}
```

#### **Delete (DELETE)**

```java
@DeleteMapping("/users/{id}")
public Mono<Void> deleteUser(@PathVariable String id) {
    return Mono.fromFuture(() -> 
        firestore.collection("users")
            .document(id)
            .delete()
            .toCompletableFuture()
    )
    .then();
}
```

#### **List (GET Collection)**

```java
@GetMapping("/users")
public Flux<User> getAllUsers() {
    return Mono.fromFuture(() -> 
        firestore.collection("users")
            .get()
            .toCompletableFuture()
    )
    .flatMapMany(querySnapshot -> 
        Flux.fromIterable(querySnapshot.getDocuments())
            .map(doc -> doc.toObject(User.class))
    );
}
```

### 🔄 Follow System (Ejemplo Completo)

```java
@Service
public class FollowService {
    
    @Autowired
    private Firestore firestore;
    
    // Seguir a un usuario (operaciones en paralelo)
    public Mono<FollowResponse> followUser(String followerId, String followingId) {
        return Mono.zip(
            // 1. Crear registro de follow
            createFollowRecord(followerId, followingId),
            // 2. Incrementar contador de followers
            incrementFollowersCount(followingId),
            // 3. Incrementar contador de following
            incrementFollowingCount(followerId)
        )
        .map(tuple -> new FollowResponse(
            "Successfully followed user",
            tuple.getT2(), // Nuevo followers count
            tuple.getT3()  // Nuevo following count
        ))
        .onErrorMap(e -> new FollowException("Failed to follow user", e));
    }
    
    private Mono<Void> createFollowRecord(String followerId, String followingId) {
        return Mono.fromFuture(() -> {
            Map<String, Object> followData = Map.of(
                "followerId", followerId,
                "followingId", followingId,
                "createdAt", FieldValue.serverTimestamp()
            );
            
            return firestore.collection("follows")
                .add(followData)
                .toCompletableFuture();
        }).then();
    }
    
    private Mono<Long> incrementFollowersCount(String userId) {
        return Mono.fromFuture(() -> 
            firestore.collection("users")
                .document(userId)
                .update("followersCount", FieldValue.increment(1))
                .toCompletableFuture()
        )
        .thenReturn(1L); // Simplificado
    }
}
```

### 📊 Consultas Complejas en Firestore (Reactivo)

```java
// Obtener posts de usuarios que sigo (feed personalizado)
public Flux<Post> getUserFeed(String userId) {
    return getFollowingIds(userId)  // Mono<List<String>>
        .flatMapMany(followingIds -> 
            Flux.fromIterable(followingIds)
                .flatMap(followingId -> 
                    getPostsByUser(followingId)  // Flux<Post>
                )
        )
        .sort(Comparator.comparing(Post::getCreatedAt).reversed())
        .take(50); // Limit
}

private Mono<List<String>> getFollowingIds(String userId) {
    return Mono.fromFuture(() -> 
        firestore.collection("follows")
            .whereEqualTo("followerId", userId)
            .get()
            .toCompletableFuture()
    )
    .map(snapshot -> 
        snapshot.getDocuments().stream()
            .map(doc -> doc.getString("followingId"))
            .collect(Collectors.toList())
    );
}

private Flux<Post> getPostsByUser(String userId) {
    return Mono.fromFuture(() -> 
        firestore.collection("posts")
            .whereEqualTo("userId", userId)
            .orderBy("createdAt", Query.Direction.DESCENDING)
            .limit(10)
            .get()
            .toCompletableFuture()
    )
    .flatMapMany(snapshot -> 
        Flux.fromIterable(snapshot.getDocuments())
            .map(doc -> doc.toObject(Post.class))
    );
}
```

### 🚀 Ventajas de WebFlux + Firebase en UPSGlam

#### **1. Alta Concurrencia**
```
Threads Tradicionales (Tomcat):    200 threads = 200 requests simultáneas
Threads Reactivos (Netty):        10 threads = 10,000+ requests simultáneas
```

#### **2. Operaciones No Bloqueantes**

```java
// ❌ BLOQUEANTE (Spring MVC tradicional)
@GetMapping("/user/{id}")
public User getUser(@PathVariable String id) {
    DocumentSnapshot snapshot = firestore
        .collection("users")
        .document(id)
        .get()
        .get();  // ⚠️ BLOQUEA EL THREAD!
    return snapshot.toObject(User.class);
}

// ✅ NO BLOQUEANTE (Spring WebFlux)
@GetMapping("/user/{id}")
public Mono<User> getUser(@PathVariable String id) {
    return Mono.fromFuture(() -> 
        firestore.collection("users")
            .document(id)
            .get()
            .toCompletableFuture()
    )
    .map(snapshot -> snapshot.toObject(User.class));
    // Thread liberado inmediatamente!
}
```

#### **3. Backpressure Natural**

Firebase Firestore tiene rate limits. WebFlux maneja automáticamente la presión:

```java
Flux<Post> posts = getAllPosts()
    .delayElements(Duration.ofMillis(100))  // Control de flujo
    .limitRate(10);  // Solicitar máximo 10 elementos a la vez
```

#### **4. Composición Elegante**

```java
// Operación compleja: Crear post + Notificar seguidores
public Mono<PostResponse> createPostAndNotify(Post post) {
    return savePost(post)  // Mono<Post>
        .flatMap(savedPost -> 
            getFollowers(post.getUserId())  // Mono<List<String>>
                .flatMapMany(Flux::fromIterable)
                .flatMap(followerId -> 
                    sendNotification(followerId, savedPost)  // Mono<Void>
                )
                .then(Mono.just(savedPost))
        )
        .map(PostResponse::new);
}
```

### 🐛 Manejo de Errores Reactivo

```java
// Error handling con recuperación
public Mono<User> getUserWithFallback(String userId) {
    return getUser(userId)
        .onErrorResume(NotFoundException.class, e -> 
            getDefaultUser()  // Fallback
        )
        .onErrorMap(FirebaseException.class, e -> 
            new ServiceException("Firebase error", e)
        )
        .doOnError(e -> log.error("Error getting user", e))
        .retry(3)  // Reintentar 3 veces
        .timeout(Duration.ofSeconds(5));  // Timeout de 5 segundos
}
```

### 📈 Performance Comparativa

| Métrica | Spring MVC (Bloqueante) | Spring WebFlux (Reactivo) |
|---------|-------------------------|---------------------------|
| Threads | 200 | 10-20 |
| Requests/seg | ~5,000 | ~50,000+ |
| Latencia P99 | 500ms | 50ms |
| Memory Usage | 2GB | 500MB |
| Escalabilidad | Vertical | Horizontal |

### 🔐 Configuración de Firebase en WebFlux

```java
@Configuration
public class FirebaseConfig {
    
    @Bean
    public FirebaseApp initializeFirebase() throws IOException {
        String credentialsPath = System.getenv("FIREBASE_CREDENTIALS_PATH");
        
        // ⚠️ IMPORTANTE: En reactive context, no usar ResourceLoader
        // Usar FileInputStream directo
        String cleanPath = credentialsPath.replace("file:", "");
        InputStream serviceAccount = new FileInputStream(cleanPath);
        
        FirebaseOptions options = FirebaseOptions.builder()
            .setCredentials(GoogleCredentials.fromStream(serviceAccount))
            .setProjectId(System.getenv("FIREBASE_PROJECT_ID"))
            .setStorageBucket(System.getenv("FIREBASE_STORAGE_BUCKET"))
            .build();
        
        return FirebaseApp.initializeApp(options);
    }
    
    @Bean
    public Firestore firestore() {
        return FirestoreClient.getFirestore();
    }
}
```

### 📝 Best Practices en UPSGlam

1. **Siempre retornar Mono/Flux**: Nunca bloquear con `.block()`
2. **Usar `flatMap` para operaciones async**: No usar `map` cuando retornas Mono/Flux
3. **Manejar errores explícitamente**: `onErrorResume`, `onErrorMap`
4. **Implementar timeouts**: `timeout(Duration.ofSeconds(5))`
5. **Logging reactivo**: `doOnNext`, `doOnError`, `doOnSuccess`
6. **Testing**: Usar `StepVerifier` de Reactor Test

### 🧪 Testing Código Reactivo

```java
@Test
void testGetUser() {
    String userId = "user123";
    User expectedUser = new User(userId, "Test User");
    
    StepVerifier.create(userService.getUser(userId))
        .expectNext(expectedUser)
        .verifyComplete();
}

@Test
void testGetUserNotFound() {
    StepVerifier.create(userService.getUser("invalid"))
        .expectError(NotFoundException.class)
        .verify();
}
```

---

## 🚀 Quick Start

```bash
# 1. Configurar credenciales
cp .env.example .env
cp docker-compose.yml.example docker-compose.yml

# 2. Editar .env con tus credenciales
notepad .env

# 3. Iniciar todos los servicios
docker-compose up -d --build

# 4. Verificar logs
docker-compose logs -f

# 5. Health checks
curl http://localhost:8080/health
curl http://localhost:8082/api/auth/health
curl http://localhost:8081/health
curl http://localhost:5000/health
```

---

## 📚 Documentación Detallada

- **[API Gateway](./api-gateway/README-DETAILED.md)** - Spring Cloud Gateway, routing, CORS
- **[Auth Service](./auth-service/README-DETAILED.md)** - WebFlux, Firebase Auth, Firestore
- **[Post Service](./post-service/README-DETAILED.md)** - R2DBC, Supabase, likes, comments
- **[CUDA Backend](./cuda-lab-back/README-DETAILED.md)** - PyCUDA, filtros GPU, kernels

---

## 👥 Equipo de Desarrollo

**UPSGlam Development Team**  
Universidad Politécnica Salesiana  
Quito, Ecuador

---

## 📄 Licencia

Proyecto académico - © 2025 Universidad Politécnica Salesiana

 

 
