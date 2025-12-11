package ec.ups.upsglam.auth.api.handler;

import ec.ups.upsglam.auth.application.FollowService;
import ec.ups.upsglam.auth.domain.dto.FollowRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.Map;

/**
 * Handler para endpoints de follow
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class FollowHandler {

    private final FollowService followService;

    /**
     * POST /follows
     * Seguir a un usuario
     */
    public Mono<ServerResponse> followUser(ServerRequest request) {
        String authHeader = request.headers().firstHeader("Authorization");
        
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return ServerResponse.status(401)
                    .bodyValue(Map.of(
                            "code", "UNAUTHORIZED",
                            "message", "Token requerido"
                    ));
        }
        
        String token = authHeader.substring(7);
        
        return request.bodyToMono(FollowRequest.class)
                .flatMap(followRequest -> followService.followUser(token, followRequest.getTargetUserId()))
                .flatMap(response -> ServerResponse.ok().bodyValue(response))
                .onErrorResume(this::handleError);
    }

    /**
     * DELETE /follows/{userId}
     * Dejar de seguir a un usuario
     */
    public Mono<ServerResponse> unfollowUser(ServerRequest request) {
        String authHeader = request.headers().firstHeader("Authorization");
        
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return ServerResponse.status(401)
                    .bodyValue(Map.of(
                            "code", "UNAUTHORIZED",
                            "message", "Token requerido"
                    ));
        }
        
        String token = authHeader.substring(7);
        String targetUserId = request.pathVariable("userId");
        
        return followService.unfollowUser(token, targetUserId)
                .flatMap(response -> ServerResponse.ok().bodyValue(response))
                .onErrorResume(this::handleError);
    }

    /**
     * GET /follows/{userId}/stats
     * Obtener estadísticas de follows de un usuario
     */
    public Mono<ServerResponse> getFollowStats(ServerRequest request) {
        String authHeader = request.headers().firstHeader("Authorization");
        
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return ServerResponse.status(401)
                    .bodyValue(Map.of(
                            "code", "UNAUTHORIZED",
                            "message", "Token requerido"
                    ));
        }
        
        String token = authHeader.substring(7);
        String targetUserId = request.pathVariable("userId");
        boolean includeList = request.queryParam("includeList")
                .map(Boolean::parseBoolean)
                .orElse(false);
        
        return followService.getFollowStats(token, targetUserId, includeList)
                .flatMap(stats -> ServerResponse.ok().bodyValue(stats))
                .onErrorResume(this::handleError);
    }

    /**
     * GET /follows/{userId}/followers
     * Obtener lista de seguidores
     */
    public Mono<ServerResponse> getFollowers(ServerRequest request) {
        String authHeader = request.headers().firstHeader("Authorization");
        
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return ServerResponse.status(401)
                    .bodyValue(Map.of(
                            "code", "UNAUTHORIZED",
                            "message", "Token requerido"
                    ));
        }
        
        String token = authHeader.substring(7);
        String targetUserId = request.pathVariable("userId");
        
        return followService.getFollowers(token, targetUserId)
                .flatMap(followers -> ServerResponse.ok().bodyValue(followers))
                .onErrorResume(this::handleError);
    }

    /**
     * GET /follows/{userId}/following
     * Obtener lista de usuarios que sigue
     */
    public Mono<ServerResponse> getFollowing(ServerRequest request) {
        String authHeader = request.headers().firstHeader("Authorization");
        
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return ServerResponse.status(401)
                    .bodyValue(Map.of(
                            "code", "UNAUTHORIZED",
                            "message", "Token requerido"
                    ));
        }
        
        String token = authHeader.substring(7);
        String targetUserId = request.pathVariable("userId");
        
        return followService.getFollowing(token, targetUserId)
                .flatMap(following -> ServerResponse.ok().bodyValue(following))
                .onErrorResume(this::handleError);
    }

    /**
     * Manejo de errores
     */
    private Mono<ServerResponse> handleError(Throwable error) {
        log.error("Error en FollowHandler: {}", error.getMessage());
        
        String code = "INTERNAL_ERROR";
        String message = error.getMessage();
        int status = 500;
        
        String errorClassName = error.getClass().getSimpleName();
        
        if (errorClassName.contains("SelfFollow")) {
            code = "SELF_FOLLOW_NOT_ALLOWED";
            status = 400;
        } else if (errorClassName.contains("AlreadyFollowing")) {
            code = "ALREADY_FOLLOWING";
            status = 409;
        } else if (errorClassName.contains("FollowNotFound")) {
            code = "FOLLOW_NOT_FOUND";
            status = 404;
        } else if (errorClassName.contains("UserNotFound")) {
            code = "USER_NOT_FOUND";
            status = 404;
        } else if (errorClassName.contains("InvalidCredentials") || errorClassName.contains("Unauthorized")) {
            code = "UNAUTHORIZED";
            status = 401;
        }
        
        return ServerResponse.status(status)
                .bodyValue(Map.of(
                        "code", code,
                        "message", message
                ));
    }
}
