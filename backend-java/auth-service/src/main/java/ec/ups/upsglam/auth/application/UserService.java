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
 * Servicio de gestion de perfil de usuario
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
        log.info(
                "updateProfile() request => username={}, fullName={}, bio={}, photoUrl={}",
                request.getUsername(),
                request.getFullName(),
                request.getBio(),
                request.getPhotoUrl()
        );
        return firebaseService.verifyToken(idToken)
                .flatMap(uid -> {
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
        if (request.getPhotoUrl() != null) {
            updates.put("photoUrl", request.getPhotoUrl());
        }

        log.info("updateUserData() uid={} updates={}", uid, updates);

        if (updates.isEmpty()) {
            return firebaseService.getUserFromFirestore(uid)
                    .map(user -> mapToUserResponse(user, true, false));
        }

        return firebaseService.updateUserInFirestore(uid, updates)
                .map(user -> mapToUserResponse(user, true, false));
    }

    /**
     * Obtener perfil publico por username, calculando si el solicitante lo sigue
     */
    public Mono<UserResponse> getUserByUsername(String username, String requesterToken) {
        Mono<String> requesterIdMono = Mono.justOrEmpty(requesterToken)
                .flatMap(firebaseService::verifyToken);

        return firebaseService.getUserByUsername(username)
                .flatMap(targetUser -> requesterIdMono
                        .switchIfEmpty(Mono.just(""))
                        .flatMap(requesterId -> {
                            boolean isMe = !requesterId.isBlank() && requesterId.equals(targetUser.getId());
                            Mono<Boolean> isFollowingMono = (!requesterId.isBlank() && !isMe)
                                    ? firebaseService.isFollowing(requesterId, targetUser.getId())
                                    : Mono.just(false);

                            return isFollowingMono
                                    .defaultIfEmpty(false)
                                    .map(isFollowing -> mapToUserResponse(targetUser, isMe, isFollowing));
                        })
                )
                .doOnSuccess(u -> log.info("Perfil publico cargado: {}", username))
                .doOnError(e -> log.error("Error perfil publico {}", username, e));
    }

    /**
     * Mapear User a UserResponse
     */
    private UserResponse mapToUserResponse(User user) {
        return mapToUserResponse(user, false, false);
    }

    private UserResponse mapToUserResponse(User user, boolean isMe, boolean isFollowing) {
        return UserResponse.builder()
                .id(user.getId())
                .email(user.getEmail())
                .username(user.getUsername())
                .fullName(user.getFullName())
                .photoUrl(user.getPhotoUrl())
                .bio(user.getBio())
                .followersCount(user.getFollowersCount())
                .followingCount(user.getFollowingCount())
                .isMe(isMe)
                .isFollowing(isFollowing)
                .build();
    }
}
