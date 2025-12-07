package ec.ups.upsglam.auth.application;

import ec.ups.upsglam.auth.domain.dto.UpdateProfileRequest;
import ec.ups.upsglam.auth.domain.dto.UserResponse;
import ec.ups.upsglam.auth.domain.exception.UsernameAlreadyInUseException;
import ec.ups.upsglam.auth.domain.model.User;
import ec.ups.upsglam.auth.infrastructure.firebase.FirebaseService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;

/**
 * Servicio de gestión de perfil de usuario
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class UserService {

    private final FirebaseService firebaseService;

    /**
     * Actualizar perfil de usuario
     */
    public Mono<UserResponse> updateProfile(String idToken, UpdateProfileRequest request) {
        return firebaseService.verifyToken(idToken)
                .flatMap(uid -> {
                    // Si cambia username, verificar que no exista
                    if (request.getUsername() != null) {
                        return firebaseService.usernameExists(request.getUsername())
                                .flatMap(exists -> {
                                    if (exists) {
                                        return Mono.error(new UsernameAlreadyInUseException(request.getUsername()));
                                    }
                                    return updateUserData(uid, request);
                                });
                    }
                    return updateUserData(uid, request);
                })
                .doOnSuccess(user -> log.info("Perfil actualizado: {}", user.getUsername()))
                .doOnError(error -> log.error("Error actualizando perfil", error));
    }

    /**
     * Actualizar datos del usuario
     */
    private Mono<UserResponse> updateUserData(String uid, UpdateProfileRequest request) {
        Map<String, Object> updates = new HashMap<>();
        
        if (request.getUsername() != null) {
            updates.put("username", request.getUsername());
        }
        if (request.getFullName() != null) {
            updates.put("fullName", request.getFullName());
        }
        if (request.getBio() != null) {
            updates.put("bio", request.getBio());
        }
        
        if (updates.isEmpty()) {
            return firebaseService.getUserFromFirestore(uid)
                    .map(this::mapToUserResponse);
        }
        
        return firebaseService.updateUserInFirestore(uid, updates)
                .map(this::mapToUserResponse);
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
}
