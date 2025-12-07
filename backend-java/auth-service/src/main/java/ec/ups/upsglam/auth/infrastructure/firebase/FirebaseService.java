package ec.ups.upsglam.auth.infrastructure.firebase;

import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseAuthException;
import com.google.firebase.auth.UserRecord;
import ec.ups.upsglam.auth.domain.exception.*;
import ec.ups.upsglam.auth.domain.model.User;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;

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
                        .build();
            } catch (InterruptedException | ExecutionException e) {
                log.error("Error obteniendo usuario de Firestore", e);
                throw new RuntimeException("Error obteniendo usuario");
            }
        }).subscribeOn(Schedulers.boundedElastic());
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
}
