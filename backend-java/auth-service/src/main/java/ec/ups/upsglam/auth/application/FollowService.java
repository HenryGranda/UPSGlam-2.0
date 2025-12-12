package ec.ups.upsglam.auth.application;

import ec.ups.upsglam.auth.domain.dto.FollowResponse;
import ec.ups.upsglam.auth.domain.dto.FollowStatsResponse;
import ec.ups.upsglam.auth.domain.dto.UserResponse;
import ec.ups.upsglam.auth.domain.exception.AlreadyFollowingException;
import ec.ups.upsglam.auth.infrastructure.firebase.FirebaseService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Servicio de gestión de follows
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class FollowService {

    private final FirebaseService firebaseService;

    /**
     * Seguir a un usuario
     */
    public Mono<FollowResponse> followUser(String idToken, String targetUserId) {
        return firebaseService.verifyToken(idToken)
                .flatMap(currentUserId ->
                        firebaseService.isFollowing(currentUserId, targetUserId)
                                .flatMap(isFollowing -> {
                                // ✅ Si ya lo sigue, NO es error (idempotente)
                                if (isFollowing) {
                                        return firebaseService.getFollowersCount(targetUserId)
                                                .map(count -> FollowResponse.builder()
                                                        .success(true)
                                                        .message("Ya sigues a este usuario")
                                                        .isFollowing(true)
                                                        .followersCount(count)
                                                        .build());
                                }

                                // ✅ Si no lo sigue, crear follow
                                return firebaseService.createFollow(currentUserId, targetUserId)
                                        .then(firebaseService.getFollowersCount(targetUserId))
                                        .map(count -> FollowResponse.builder()
                                                .success(true)
                                                .message("Ahora sigues a este usuario")
                                                .isFollowing(true)
                                                .followersCount(count)
                                                .build());
                                })
                )
                .doOnSuccess(response -> log.info("Follow OK: {}", response.getMessage()))
                .doOnError(error -> log.error("Error al seguir usuario", error));
    }

    /**
     * Dejar de seguir a un usuario
     */
    public Mono<FollowResponse> unfollowUser(String idToken, String targetUserId) {
        return firebaseService.verifyToken(idToken)
                .flatMap(currentUserId -> 
                    firebaseService.deleteFollow(currentUserId, targetUserId)
                            .then(firebaseService.getFollowersCount(targetUserId))
                            .map(count -> FollowResponse.builder()
                                    .success(true)
                                    .message("Dejaste de seguir a este usuario")
                                    .isFollowing(false)
                                    .followersCount(count)
                                    .build())
                )
                .doOnSuccess(response -> log.info("Unfollow exitoso"))
                .doOnError(error -> log.error("Error al dejar de seguir usuario", error));
    }

    /**
     * Obtener estadísticas de follows de un usuario
     */
    public Mono<FollowStatsResponse> getFollowStats(String idToken, String targetUserId, boolean includeList) {
        return firebaseService.verifyToken(idToken)
                .flatMap(currentUserId -> {
                    // Obtener conteos
                    Mono<Long> followersMono = firebaseService.getFollowersCount(targetUserId);
                    Mono<Long> followingMono = firebaseService.getFollowingCount(targetUserId);
                    Mono<Boolean> isFollowingMono = currentUserId.equals(targetUserId) 
                            ? Mono.just(false) 
                            : firebaseService.isFollowing(currentUserId, targetUserId);

                    if (includeList) {
                        // Incluir listas de usuarios
                        Mono<List<UserResponse>> followersList = firebaseService.getFollowers(targetUserId)
                                .map(users -> users.stream()
                                        .map(this::mapToUserResponse)
                                        .collect(Collectors.toList()));
                        
                        Mono<List<UserResponse>> followingList = firebaseService.getFollowing(targetUserId)
                                .map(users -> users.stream()
                                        .map(this::mapToUserResponse)
                                        .collect(Collectors.toList()));

                        return Mono.zip(followersMono, followingMono, isFollowingMono, followersList, followingList)
                                .map(tuple -> FollowStatsResponse.builder()
                                        .userId(targetUserId)
                                        .followersCount(tuple.getT1())
                                        .followingCount(tuple.getT2())
                                        .isFollowing(currentUserId.equals(targetUserId) ? null : tuple.getT3())
                                        .followers(tuple.getT4())
                                        .following(tuple.getT5())
                                        .build());
                    } else {
                        // Solo conteos
                        return Mono.zip(followersMono, followingMono, isFollowingMono)
                                .map(tuple -> FollowStatsResponse.builder()
                                        .userId(targetUserId)
                                        .followersCount(tuple.getT1())
                                        .followingCount(tuple.getT2())
                                        .isFollowing(currentUserId.equals(targetUserId) ? null : tuple.getT3())
                                        .build());
                    }
                })
                .doOnSuccess(stats -> log.info("Estadísticas de follow obtenidas para: {}", targetUserId))
                .doOnError(error -> log.error("Error obteniendo estadísticas de follow", error));
    }

    /**
     * Obtener lista de seguidores
     */
    public Mono<List<UserResponse>> getFollowers(String idToken, String targetUserId) {
        return firebaseService.verifyToken(idToken)
                .flatMap(currentUserId -> firebaseService.getFollowers(targetUserId))
                .map(users -> users.stream()
                        .map(this::mapToUserResponse)
                        .collect(Collectors.toList()))
                .doOnError(error -> log.error("Error obteniendo seguidores", error));
    }

    /**
     * Obtener lista de usuarios que sigue
     */
    public Mono<List<UserResponse>> getFollowing(String idToken, String targetUserId) {
        return firebaseService.verifyToken(idToken)
                .flatMap(currentUserId -> firebaseService.getFollowing(targetUserId))
                .map(users -> users.stream()
                        .map(this::mapToUserResponse)
                        .collect(Collectors.toList()))
                .doOnError(error -> log.error("Error obteniendo following", error));
    }

    /**
     * Mapear User a UserResponse
     */
      @SuppressWarnings("unchecked")
      private UserResponse mapToUserResponse(Object raw) {

        if (raw instanceof java.util.Map<?, ?> map) {
                return UserResponse.builder()
                        .id(String.valueOf(map.get("id")))
                        .email(map.get("email") != null ? map.get("email").toString() : "")
                        .username(map.get("username") != null ? map.get("username").toString() : "")
                        .fullName(map.get("fullName") != null ? map.get("fullName").toString() : "")
                        .photoUrl(map.get("photoUrl") != null ? map.get("photoUrl").toString() : null)
                        .bio(map.get("bio") != null ? map.get("bio").toString() : null)
                        .followersCount(
                        map.get("followersCount") instanceof Number
                                ? ((Number) map.get("followersCount")).longValue()
                                : 0L
                        )
                        .followingCount(
                        map.get("followingCount") instanceof Number
                                ? ((Number) map.get("followingCount")).longValue()
                                : 0L
                        )
                        .build();
        }

        throw new IllegalArgumentException("Formato de usuario no soportado: " + raw);
        }

}
