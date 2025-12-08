package ec.ups.upsglam.post.api.handler;

import ec.ups.upsglam.post.application.service.LikeService;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.Map;

@Component
@RequiredArgsConstructor
public class LikeHandler {

    private static final Logger log = LoggerFactory.getLogger(LikeHandler.class);
    private final LikeService likeService;

    /**
     * POST /posts/{postId}/likes
     * Dar like a un post
     */
    public Mono<ServerResponse> likePost(ServerRequest request) {
        String postId = request.pathVariable("postId");
        String userId = extractUserId(request);
        String username = extractUsername(request);

        log.info("User {} liking post {}", userId, postId);

        return likeService.likePost(postId, userId)
                .flatMap(likeResponse ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(likeResponse)
                )
                .onErrorResume(this::handleError);
    }

    /**
     * DELETE /posts/{postId}/likes
     * Quitar like de un post
     */
    public Mono<ServerResponse> unlikePost(ServerRequest request) {
        String postId = request.pathVariable("postId");
        String userId = extractUserId(request);

        log.info("User {} unliking post {}", userId, postId);

        return likeService.unlikePost(postId, userId)
                .flatMap(likeResponse ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(likeResponse)
                )
                .onErrorResume(this::handleError);
    }

    /**
     * GET /posts/{postId}/likes?page=0&size=20
     * Obtener lista de usuarios que dieron like
     */
    public Mono<ServerResponse> getPostLikes(ServerRequest request) {
        String postId = request.pathVariable("postId");
        int page = Integer.parseInt(request.queryParam("page").orElse("0"));
        int size = Integer.parseInt(request.queryParam("size").orElse("20"));

        log.info("Getting likes for post: {}, page: {}, size: {}", postId, page, size);

        // Por ahora retornamos lista vacía, este método requiere implementación adicional
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "postId", postId,
                        "page", page,
                        "size", size,
                        "items", java.util.Collections.emptyList(),
                        "totalItems", 0
                ))
                .onErrorResume(this::handleError);
    }

    /**
     * Extrae el userId del header X-User-Id
     */
    private String extractUserId(ServerRequest request) {
        return request.headers()
                .firstHeader("X-User-Id");
    }

    /**
     * Extrae el username del header X-Username
     */
    private String extractUsername(ServerRequest request) {
        return request.headers()
                .firstHeader("X-Username");
    }

    /**
     * Manejo centralizado de errores
     */
    private Mono<ServerResponse> handleError(Throwable error) {
        log.error("Error in LikeHandler: ", error);
        return ServerResponse.status(500)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "error", "INTERNAL_SERVER_ERROR",
                        "message", error.getMessage()
                ));
    }
}
