package ec.ups.upsglam.auth.infrastructure.firebase;

import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseAuthException;
import com.google.firebase.auth.UserRecord;
import ec.ups.upsglam.auth.domain.exception.*;
import ec.ups.upsglam.auth.domain.model.Follow;
import ec.ups.upsglam.auth.domain.model.User;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

/**
 * Servicio para interactuar con Firebase Auth y Firestore
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class FirebaseService {

    private final FirebaseAuth firebaseAuth;
    private final Firestore firestore;
    
    private static final String USERS_COLLECTION = "users";
    private static final String FOLLOWS_COLLECTION = "follows";
    private static final java.util.Random RANDOM = new java.util.Random();

    /**
     * Crear usuario en Firebase Auth
     */
    public Mono<UserRecord> createAuthUser(String email, String password) {
        return Mono.fromCallable(() -> {
            try {
                UserRecord.CreateRequest request = new UserRecord.CreateRequest()
                        .setEmail(email)
                        .setPassword(password)
                        .setEmailVerified(false);
                
                UserRecord userRecord = firebaseAuth.createUser(request);
                log.info("Usuario creado en Firebase Auth: {}", userRecord.getUid());
                return userRecord;
            } catch (FirebaseAuthException e) {
                if (e.getMessage().contains("EMAIL_EXISTS")) {
                    throw new EmailAlreadyInUseException(email);
                }
                log.error("Error creando usuario en Firebase Auth", e);
                throw new RuntimeException("Error al crear usuario: " + e.getMessage());
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Verificar si el username ya existe en Firestore
     */
    public Mono<Boolean> usernameExists(String username) {
        return Mono.fromCallable(() -> {
            try {
                var query = firestore.collection(USERS_COLLECTION)
                        .whereEqualTo("username", username)
                        .limit(1)
                        .get();
                
                return !query.get().isEmpty();
            } catch (Exception e) {
                // Si la colección no existe, el username no existe
                log.debug("Colección no existe o error verificando username, asumiendo false: {}", e.getMessage());
                return false;
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Guardar datos de usuario en Firestore
     */
    public Mono<User> saveUserToFirestore(String uid, String email, String username, String fullName) {
        return Mono.fromCallable(() -> {
            try {
                Map<String, Object> userData = new HashMap<>();
                userData.put("email", email);
                userData.put("username", username);
                userData.put("fullName", fullName);
                userData.put("photoUrl", null);
                userData.put("bio", null);
                userData.put("createdAt", System.currentTimeMillis());
                userData.put("followersCount", 0L);
                userData.put("followingCount", 0L);
                
                firestore.collection(USERS_COLLECTION)
                        .document(uid)
                        .set(userData)
                        .get();
                
                log.info("Usuario guardado en Firestore: {}", uid);
                
                return User.builder()
                        .id(uid)
                        .email(email)
                        .username(username)
                        .fullName(fullName)
                        .photoUrl(null)
                        .bio(null)
                        .createdAt((Long) userData.get("createdAt"))
                        .followersCount(0L)
                        .followingCount(0L)
                        .build();
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error guardando usuario en Firestore", e);
                throw new RuntimeException("Error guardando usuario");
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Obtener usuario de Firestore por UID
     */
    public Mono<User> getUserFromFirestore(String uid) {
        return Mono.fromCallable(() -> {
            try {
                DocumentSnapshot document = firestore.collection(USERS_COLLECTION)
                        .document(uid)
                        .get()
                        .get();
                
                if (!document.exists()) {
                    throw new UserNotFoundException(uid);
                }
                
                return User.builder()
                        .id(document.getId())
                        .email(document.getString("email"))
                        .username(document.getString("username"))
                        .fullName(document.getString("fullName"))
                        .photoUrl(document.getString("photoUrl"))
                        .bio(document.getString("bio"))
                        .createdAt(document.getLong("createdAt"))
                        .followersCount(document.getLong("followersCount") != null ? document.getLong("followersCount") : 0L)
                        .followingCount(document.getLong("followingCount") != null ? document.getLong("followingCount") : 0L)
                        .build();
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error obteniendo usuario de Firestore", e);
                throw new RuntimeException("Error obteniendo usuario");
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Obtener usuario; si no existe en Firestore (login social primera vez), lo crea usando Firebase Auth.
     */
    public Mono<User> getOrCreateUser(String uid) {
        return getUserFromFirestore(uid)
                .onErrorResume(UserNotFoundException.class, e -> Mono.fromCallable(() -> {
                    try {
                        UserRecord record = firebaseAuth.getUser(uid);
                        String email = record.getEmail();
                        String fullName = record.getDisplayName();
                        String photoUrl = record.getPhotoUrl();

                        String username = generateUniqueUsername(email, uid);

                        Map<String, Object> userData = new HashMap<>();
                        userData.put("email", email);
                        userData.put("username", username);
                        userData.put("fullName", fullName);
                        userData.put("photoUrl", photoUrl);
                        userData.put("bio", null);
                        userData.put("createdAt", System.currentTimeMillis());
                        userData.put("followersCount", 0L);
                        userData.put("followingCount", 0L);

                        firestore.collection(USERS_COLLECTION)
                                .document(uid)
                                .set(userData)
                                .get();

                        log.info("Perfil creado en Firestore para uid={} (login social)", uid);

                        return User.builder()
                                .id(uid)
                                .email(email)
                                .username(username)
                                .fullName(fullName)
                                .photoUrl(photoUrl)
                                .bio(null)
                                .createdAt((Long) userData.get("createdAt"))
                                .followersCount(0L)
                                .followingCount(0L)
                                .build();
                    } catch (Exception ex) {
                        log.error("Error creando perfil para uid={} desde Firebase Auth", uid, ex);
                        throw new RuntimeException("Error creando perfil para usuario social");
                    }
                }).subscribeOn(Schedulers.boundedElastic()));
    }

    /**
     * Buscar usuario por email en Firestore
     */
    public Mono<User> getUserByEmail(String email) {
        return Mono.fromCallable(() -> {
            try {
                var query = firestore.collection(USERS_COLLECTION)
                        .whereEqualTo("email", email)
                        .limit(1)
                        .get();
                
                var documents = query.get().getDocuments();
                if (documents.isEmpty()) {
                    throw new UserNotFoundException(email);
                }
                
                DocumentSnapshot document = documents.get(0);
                return User.builder()
                        .id(document.getId())
                        .email(document.getString("email"))
                        .username(document.getString("username"))
                        .fullName(document.getString("fullName"))
                        .photoUrl(document.getString("photoUrl"))
                        .bio(document.getString("bio"))
                        .createdAt(document.getLong("createdAt"))
                        .followersCount(document.getLong("followersCount") != null ? document.getLong("followersCount") : 0L)
                        .followingCount(document.getLong("followingCount") != null ? document.getLong("followingCount") : 0L)
                        .build();
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error buscando usuario por email", e);
                throw new UserNotFoundException(email);
            } catch (Exception e) {
                log.error("Error inesperado buscando usuario por email", e);
                throw new UserNotFoundException(email);
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Buscar usuario por username en Firestore
     */
    public Mono<User> getUserByUsername(String username) {
        return Mono.fromCallable(() -> {
            try {
                var query = firestore.collection(USERS_COLLECTION)
                        .whereEqualTo("username", username)
                        .limit(1)
                        .get();
                
                var documents = query.get().getDocuments();
                if (documents.isEmpty()) {
                    throw new UserNotFoundException(username);
                }
                
                DocumentSnapshot document = documents.get(0);
                return User.builder()
                        .id(document.getId())
                        .email(document.getString("email"))
                        .username(document.getString("username"))
                        .fullName(document.getString("fullName"))
                        .photoUrl(document.getString("photoUrl"))
                        .bio(document.getString("bio"))
                        .createdAt(document.getLong("createdAt"))
                        .followersCount(document.getLong("followersCount") != null ? document.getLong("followersCount") : 0L)
                        .followingCount(document.getLong("followingCount") != null ? document.getLong("followingCount") : 0L)
                        .build();
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error buscando usuario por username", e);
                throw new RuntimeException("Error buscando usuario");
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Actualizar datos de usuario en Firestore
     */
    public Mono<User> updateUserInFirestore(String uid, Map<String, Object> updates) {
        return Mono.fromCallable(() -> {
            try {
                firestore.collection(USERS_COLLECTION)
                        .document(uid)
                        .update(updates)
                        .get();
                
                log.info("Usuario actualizado en Firestore: {}", uid);
                return getUserFromFirestore(uid).block();
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error actualizando usuario en Firestore", e);
                throw new RuntimeException("Error actualizando usuario");
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Verificar token de Firebase
     */
    public Mono<String> verifyToken(String idToken) {
        return Mono.fromCallable(() -> {
            try {
                var decodedToken = firebaseAuth.verifyIdToken(idToken);
                return decodedToken.getUid();
            } catch (FirebaseAuthException e) {
                log.error("Token inválido o expirado", e);
                throw new InvalidCredentialsException();
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Obtener usuario de Firebase Auth por email
     */
    public Mono<UserRecord> getAuthUserByEmail(String email) {
        return Mono.fromCallable(() -> {
            try {
                return firebaseAuth.getUserByEmail(email);
            } catch (FirebaseAuthException e) {
                throw new UserNotFoundException(email);
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Crear custom token para login
     */
    public Mono<String> createCustomToken(String uid) {
        return Mono.fromCallable(() -> {
            try {
                return firebaseAuth.createCustomToken(uid);
            } catch (FirebaseAuthException e) {
                log.error("Error creando custom token", e);
                throw new RuntimeException("Error creando token");
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    // ==================== MÉTODOS PARA FOLLOWS ====================

    /**
     * Crear una relación de follow en Firestore
     */
    public Mono<Follow> createFollow(String followerUserId, String followedUserId) {
        return Mono.fromCallable(() -> {
            try {
                // Validar que no sea auto-follow
                if (followerUserId.equals(followedUserId)) {
                    throw new SelfFollowException();
                }

                // Verificar que ambos usuarios existen
                if (!userExists(followerUserId) || !userExists(followedUserId)) {
                    throw new UserNotFoundException("Uno de los usuarios no existe");
                }

                // Crear el documento de follow
                String followId = followerUserId + "_" + followedUserId;
                Map<String, Object> followData = new HashMap<>();
                followData.put("followerUserId", followerUserId);
                followData.put("followedUserId", followedUserId);
                followData.put("createdAt", System.currentTimeMillis());

                firestore.collection(FOLLOWS_COLLECTION)
                        .document(followId)
                        .set(followData)
                        .get();

                // Incrementar contadores
                incrementFollowersCount(followedUserId);
                incrementFollowingCount(followerUserId);

                log.info("Follow creado: {} sigue a {}", followerUserId, followedUserId);

                return Follow.builder()
                        .id(followId)
                        .followerUserId(followerUserId)
                        .followedUserId(followedUserId)
                        .createdAt((Long) followData.get("createdAt"))
                        .build();
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error creando follow", e);
                throw new RuntimeException("Error creando follow");
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Eliminar una relación de follow
     */
    public Mono<Void> deleteFollow(String followerUserId, String followedUserId) {
        return Mono.fromCallable(() -> {
            try {
                String followId = followerUserId + "_" + followedUserId;
                
                // Verificar que existe el follow
                var doc = firestore.collection(FOLLOWS_COLLECTION)
                        .document(followId)
                        .get()
                        .get();

                if (!doc.exists()) {
                    throw new FollowNotFoundException(followerUserId, followedUserId);
                }

                // Eliminar el documento
                firestore.collection(FOLLOWS_COLLECTION)
                        .document(followId)
                        .delete()
                        .get();

                // Decrementar contadores
                decrementFollowersCount(followedUserId);
                decrementFollowingCount(followerUserId);

                log.info("Follow eliminado: {} dejó de seguir a {}", followerUserId, followedUserId);
                return null;
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error eliminando follow", e);
                throw new RuntimeException("Error eliminando follow");
            }
        }).subscribeOn(Schedulers.boundedElastic()).then();
    }

    /**
     * Verificar si un usuario sigue a otro
     */
    public Mono<Boolean> isFollowing(String followerUserId, String followedUserId) {
        return Mono.fromCallable(() -> {
            try {
                String followId = followerUserId + "_" + followedUserId;
                var doc = firestore.collection(FOLLOWS_COLLECTION)
                        .document(followId)
                        .get()
                        .get();
                return doc.exists();
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error verificando follow", e);
                return false;
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Obtener lista de seguidores de un usuario
     */
    public Mono<List<User>> getFollowers(String userId) {
        return Mono.fromCallable(() -> {
            try {
                var query = firestore.collection(FOLLOWS_COLLECTION)
                        .whereEqualTo("followedUserId", userId)
                        .get()
                        .get();

                return query.getDocuments().stream()
                        .map(doc -> doc.getString("followerUserId"))
                        .map(followerId -> getUserFromFirestore(followerId).block())
                        .collect(Collectors.toList());
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error obteniendo seguidores", e);
                throw new RuntimeException("Error obteniendo seguidores");
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Obtener lista de usuarios que sigue un usuario
     */
    public Mono<List<User>> getFollowing(String userId) {
        return Mono.fromCallable(() -> {
            try {
                var query = firestore.collection(FOLLOWS_COLLECTION)
                        .whereEqualTo("followerUserId", userId)
                        .get()
                        .get();

                return query.getDocuments().stream()
                        .map(doc -> doc.getString("followedUserId"))
                        .map(followedId -> getUserFromFirestore(followedId).block())
                        .collect(Collectors.toList());
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error obteniendo following", e);
                throw new RuntimeException("Error obteniendo following");
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Obtener conteo de seguidores
     */
    public Mono<Long> getFollowersCount(String userId) {
        return Mono.fromCallable(() -> {
            try {
                var user = getUserFromFirestore(userId).block();
                return user.getFollowersCount() != null ? user.getFollowersCount() : 0L;
            } catch (Exception e) {
                log.error("Error obteniendo conteo de seguidores", e);
                return 0L;
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * Obtener conteo de following
     */
    public Mono<Long> getFollowingCount(String userId) {
        return Mono.fromCallable(() -> {
            try {
                var user = getUserFromFirestore(userId).block();
                return user.getFollowingCount() != null ? user.getFollowingCount() : 0L;
            } catch (Exception e) {
                log.error("Error obteniendo conteo de following", e);
                return 0L;
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    // ==================== MÉTODOS PRIVADOS AUXILIARES ====================

    /**
     * Verificar si un usuario existe
     */
    private boolean userExists(String userId) {
        try {
            var doc = firestore.collection(USERS_COLLECTION)
                    .document(userId)
                    .get()
                    .get();
            return doc.exists();
        } catch (InterruptedException | ExecutionException e) {
            return false;
        }
    }

    /**
     * Verificar si un username existe (llamada síncrona para generación automática).
     */
    private boolean usernameExistsSync(String username) {
        try {
            var query = firestore.collection(USERS_COLLECTION)
                    .whereEqualTo("username", username)
                    .limit(1)
                    .get()
                    .get();
            return !query.isEmpty();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Generar un username único basado en email o uid.
     */
    private String generateUniqueUsername(String email, String uid) {
        String base;
        if (email != null && email.contains("@")) {
            base = email.substring(0, email.indexOf('@'));
        } else {
            base = "user" + uid.substring(0, Math.min(uid.length(), 6));
        }
        base = base.replaceAll("[^a-zA-Z0-9_]", "").toLowerCase();
        if (base.isBlank()) base = "user";

        String candidate = base;
        int attempts = 0;
        while (usernameExistsSync(candidate) && attempts < 10) {
            candidate = base + RANDOM.nextInt(9000) + 1000; // 4 dígitos
            attempts++;
        }
        return candidate;
    }

    /**
     * Incrementar contador de seguidores
     */
    private void incrementFollowersCount(String userId) {
        try {
            var userDoc = firestore.collection(USERS_COLLECTION).document(userId);
            var doc = userDoc.get().get();
            Long currentCount = doc.getLong("followersCount");
            currentCount = currentCount != null ? currentCount : 0L;
            
            Map<String, Object> update = new HashMap<>();
            update.put("followersCount", currentCount + 1);
            userDoc.update(update).get();
        } catch (Exception e) {
            log.error("Error incrementando followersCount", e);
        }
    }

    /**
     * Decrementar contador de seguidores
     */
    private void decrementFollowersCount(String userId) {
        try {
            var userDoc = firestore.collection(USERS_COLLECTION).document(userId);
            var doc = userDoc.get().get();
            Long currentCount = doc.getLong("followersCount");
            currentCount = currentCount != null ? currentCount : 0L;
            
            Map<String, Object> update = new HashMap<>();
            update.put("followersCount", Math.max(0, currentCount - 1));
            userDoc.update(update).get();
        } catch (Exception e) {
            log.error("Error decrementando followersCount", e);
        }
    }

    /**
     * Incrementar contador de following
     */
    private void incrementFollowingCount(String userId) {
        try {
            var userDoc = firestore.collection(USERS_COLLECTION).document(userId);
            var doc = userDoc.get().get();
            Long currentCount = doc.getLong("followingCount");
            currentCount = currentCount != null ? currentCount : 0L;
            
            Map<String, Object> update = new HashMap<>();
            update.put("followingCount", currentCount + 1);
            userDoc.update(update).get();
        } catch (Exception e) {
            log.error("Error incrementando followingCount", e);
        }
    }

    /**
     * Decrementar contador de following
     */
    private void decrementFollowingCount(String userId) {
        try {
            var userDoc = firestore.collection(USERS_COLLECTION).document(userId);
            var doc = userDoc.get().get();
            Long currentCount = doc.getLong("followingCount");
            currentCount = currentCount != null ? currentCount : 0L;
            
            Map<String, Object> update = new HashMap<>();
            update.put("followingCount", Math.max(0, currentCount - 1));
            userDoc.update(update).get();
        } catch (Exception e) {
            log.error("Error decrementando followingCount", e);
        }
    }
}
