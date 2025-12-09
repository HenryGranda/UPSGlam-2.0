package ec.ups.upsglam.auth.application;

import com.google.firebase.auth.UserRecord;
import ec.ups.upsglam.auth.domain.dto.*;
import ec.ups.upsglam.auth.domain.exception.*;
import ec.ups.upsglam.auth.domain.model.User;
import ec.ups.upsglam.auth.infrastructure.firebase.FirebaseAuthRestClient;
import ec.ups.upsglam.auth.infrastructure.firebase.FirebaseService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;

/**
 * Servicio de autenticación
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AuthService {

    private final FirebaseService firebaseService;
    private final FirebaseAuthRestClient firebaseAuthRestClient;

    /**
     * Registrar nuevo usuario
     */
    public Mono<AuthResponse> register(RegisterRequest request) {
        log.info("Registrando usuario: {}", request.getEmail());

        // 1. Verificar que el username no exista
        return firebaseService.usernameExists(request.getUsername())
                .flatMap(exists -> {
                    if (exists) {
                        return Mono.error(new UsernameAlreadyInUseException(request.getUsername()));
                    }
                    
                    // 2. Crear usuario en Firebase Auth
                    return firebaseService.createAuthUser(request.getEmail(), request.getPassword());
                })
                .flatMap(userRecord -> {
                    // 3. Guardar datos adicionales en Firestore
                    return firebaseService.saveUserToFirestore(
                            userRecord.getUid(),
                            request.getEmail(),
                            request.getUsername(),
                            request.getFullName()
                    ).map(user -> Map.entry(userRecord, user));
                })
                .flatMap(entry -> {
                    UserRecord userRecord = entry.getKey();
                    User user = entry.getValue();
                    
                    // 4. Crear token personalizado
                    return firebaseService.createCustomToken(userRecord.getUid())
                            .map(token -> buildAuthResponse(user, token));
                })
                .doOnSuccess(response -> log.info("Usuario registrado exitosamente: {}", request.getUsername()))
                .doOnError(error -> log.error("Error registrando usuario", error));
    }

    /**
     * Login de usuario
     * Usa la REST API de Firebase para autenticar y obtener un ID Token real
     */
    public Mono<AuthResponse> login(LoginRequest request) {
        log.info("Intentando login: {}", request.getIdentifier());

        // Determinar si es email o username
        return (isEmail(request.getIdentifier()) 
                ? Mono.just(request.getIdentifier())
                : firebaseService.getUserByUsername(request.getIdentifier())
                        .map(User::getEmail))
                .flatMap(email -> 
                    // Autenticar con Firebase usando REST API (obtiene ID Token real)
                    firebaseAuthRestClient.signInWithPassword(email, request.getPassword())
                            .flatMap(authResponse -> 
                                // Obtener usuario de Firestore
                                firebaseService.getUserFromFirestore(authResponse.getLocalId())
                                        .map(user -> AuthResponse.builder()
                                                .user(mapToUserResponse(user))
                                                .token(TokenResponse.builder()
                                                        .idToken(authResponse.getIdToken())
                                                        .refreshToken(authResponse.getRefreshToken())
                                                        .expiresIn(Long.parseLong(authResponse.getExpiresIn()))
                                                        .build())
                                                .build())
                            )
                )
                .doOnSuccess(response -> log.info("Login exitoso: {}", request.getIdentifier()))
                .doOnError(error -> log.error("Error en login", error))
                .onErrorMap(e -> {
                    if (e instanceof UserNotFoundException) {
                        return new InvalidCredentialsException();
                    }
                    return new InvalidCredentialsException();
                });
    }

    /**
     * Obtener perfil del usuario autenticado
     */
    public Mono<UserResponse> getMe(String idToken) {
        return firebaseService.verifyToken(idToken)
                .flatMap(firebaseService::getUserFromFirestore)
                .map(this::mapToUserResponse)
                .doOnSuccess(user -> log.info("Perfil obtenido: {}", user.getUsername()))
                .doOnError(error -> log.error("Error obteniendo perfil", error));
    }

    /**
     * Construir respuesta de autenticación
     */
    private AuthResponse buildAuthResponse(User user, String customToken) {
        return AuthResponse.builder()
                .user(mapToUserResponse(user))
                .token(TokenResponse.builder()
                        .idToken(customToken)
                        .refreshToken(null)  // Firebase maneja refresh automáticamente
                        .expiresIn(3600L)
                        .build())
                .build();
    }

    /**
     * Mapear User a UserResponse
     */
    private UserResponse mapToUserResponse(User user) {
        return UserResponse.builder()
                .id(user.getId())
                .email(user.getEmail())
                .username(user.getUsername())
                .fullName(user.getFullName())
                .photoUrl(user.getPhotoUrl())
                .bio(user.getBio())
                .build();
    }

    /**
     * Verificar si es email
     */
    private boolean isEmail(String identifier) {
        return identifier.contains("@");
    }
}
