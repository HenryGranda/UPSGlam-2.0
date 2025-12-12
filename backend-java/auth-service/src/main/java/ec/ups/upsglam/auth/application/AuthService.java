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

        return firebaseService.usernameExists(request.getUsername())
                .flatMap(exists -> {
                    if (exists) {
                        return Mono.error(new UsernameAlreadyInUseException(request.getUsername()));
                    }
                    return firebaseService.createAuthUser(
                            request.getEmail(),
                            request.getPassword()
                    );
                })
                .flatMap(userRecord ->
                        firebaseService.saveUserToFirestore(
                                userRecord.getUid(),
                                request.getEmail(),
                                request.getUsername(),
                                request.getFullName()
                        ).map(user -> Map.entry(userRecord, user))
                )
                .flatMap(entry ->
                        firebaseService.createCustomToken(entry.getKey().getUid())
                                .map(token -> buildAuthResponse(entry.getValue(), token))
                )
                .doOnSuccess(r -> log.info("Usuario registrado: {}", request.getUsername()))
                .doOnError(e -> log.error("Error registrando usuario", e));
    }

    /**
     * Login de usuario
     */
    public Mono<AuthResponse> login(LoginRequest request) {
        log.info("Intentando login: {}", request.getIdentifier());

        return (isEmail(request.getIdentifier())
                ? Mono.just(request.getIdentifier())
                : firebaseService.getUserByUsername(request.getIdentifier())
                        .map(User::getEmail))
                .flatMap(email ->
                        firebaseAuthRestClient
                                .signInWithPassword(email, request.getPassword())
                                .flatMap(authResponse ->
                                        firebaseService
                                                .getUserFromFirestore(authResponse.getLocalId())
                                                .map(user ->
                                                        AuthResponse.builder()
                                                                .user(mapToUserResponse(user))
                                                                .token(TokenResponse.builder()
                                                                        .idToken(authResponse.getIdToken())
                                                                        .refreshToken(authResponse.getRefreshToken())
                                                                        .expiresIn(Long.parseLong(authResponse.getExpiresIn()))
                                                                        .build()
                                                                )
                                                                .build()
                                                )
                                )
                )
                .doOnSuccess(r -> log.info("Login exitoso"))
                .onErrorMap(e -> new InvalidCredentialsException());
    }

    /**
     * Obtener perfil del usuario autenticado
     */
    public Mono<UserResponse> getMe(String idToken) {
        return firebaseService.verifyToken(idToken)
                // Si el usuario no existe en Firestore (login social por primera vez),
                // lo creamos con los datos de Firebase Auth.
                .flatMap(firebaseService::getOrCreateUser)
                .map(this::mapToUserResponse)
                .doOnSuccess(u -> log.info("Perfil cargado: {}", u.getUsername()));
    }

    /**
     * Construir AuthResponse
     */
    private AuthResponse buildAuthResponse(User user, String token) {
        return AuthResponse.builder()
                .user(mapToUserResponse(user))
                .token(TokenResponse.builder()
                        .idToken(token)
                        .refreshToken(null)
                        .expiresIn(3600L)
                        .build())
                .build();
    }


    private UserResponse mapToUserResponse(User user) {
        return UserResponse.builder()
                .id(user.getId())
                .email(user.getEmail())
                .username(user.getUsername())
                .fullName(user.getFullName())
                .photoUrl(user.getPhotoUrl())
                .bio(user.getBio())
                .followersCount(user.getFollowersCount())
                .followingCount(user.getFollowingCount())
                .isMe(true)
                .isFollowing(false)
                .build();
    }

    private boolean isEmail(String identifier) {
        return identifier.contains("@");
    }
}
