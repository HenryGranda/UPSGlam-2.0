package ec.ups.upsglam.post.api.handler;

import ec.ups.upsglam.post.application.service.PostService;
import ec.ups.upsglam.post.domain.post.dto.CreatePostRequest;
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
public class PostHandler {

    private static final Logger log = LoggerFactory.getLogger(PostHandler.class);
    private final PostService postService;

    /**
     * GET /feed?page=0&size=10
     * Obtiene el feed paginado de posts
     */
    public Mono<ServerResponse> getFeed(ServerRequest request) {
        int page = Integer.parseInt(request.queryParam("page").orElse("0"));
        int size = Integer.parseInt(request.queryParam("size").orElse("10"));
        String userId = extractUserId(request);

        log.info("Getting feed - page: {}, size: {}, userId: {}", page, size, userId);

        return postService.getUserFeed(userId, size)
                .flatMap(feedResponse ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(feedResponse)
                )
                .onErrorResume(this::handleError);
    }

    /**
     * GET /posts/{postId}
     * Obtiene detalle de un post especifico
     */
    public Mono<ServerResponse> getPostById(ServerRequest request) {
        String postId = request.pathVariable("postId");
        String userId = extractUserId(request);

        log.info("Getting post by id: {}, userId: {}", postId, userId);

        return postService.getPostById(postId, userId)
                .flatMap(postResponse ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(postResponse)
                )
                .onErrorResume(this::handleError);
    }

    /**
     * GET /posts/user/{userIdOrUsername}
     * Obtiene posts de un usuario (acepta userId o username)
     */
    public Mono<ServerResponse> getUserPosts(ServerRequest request) {
        String userRef = request.pathVariable("userId");
        int page = Integer.parseInt(request.queryParam("page").orElse("0"));
        int size = Integer.parseInt(request.queryParam("size").orElse("10"));
        String currentUserId = extractUserId(request);

        log.info("Getting posts for userRef: {}, page: {}, size: {}, requester: {}", userRef, page, size, currentUserId);

        return postService.getUserPosts(userRef, page, size, currentUserId)
                .flatMap(feed -> ServerResponse.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(feed))
                .onErrorResume(this::handleError);
    }

    /**
     * POST /posts
     * Crea un nuevo post (con tempImageId previamente procesado)
     */
    public Mono<ServerResponse> createPost(ServerRequest request) {
        String userId = extractUserId(request);

        log.info("Creating post for user: {}", userId);

        return request.bodyToMono(CreatePostRequest.class)
                .flatMap(createRequest -> postService.createPost(createRequest, userId))
                .flatMap(postResponse ->
                        ServerResponse.status(201)
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(postResponse)
                )
                .onErrorResume(this::handleError);
    }

    /**
     * DELETE /posts/{postId}
     * Elimina un post (solo el dueno)
     */
    public Mono<ServerResponse> deletePost(ServerRequest request) {
        String postId = request.pathVariable("postId");
        String userId = extractUserId(request);
        String username = extractUsername(request);

        log.info("Deleting post: {}, userId: {}, username: {}", postId, userId, username);

        return postService.deletePost(postId, userId, username)
                .then(ServerResponse.noContent().build())
                .onErrorResume(this::handleError);
    }

    /**
     * PATCH /posts/{postId}/caption
     * Actualiza la descripcion de un post
     */
    public Mono<ServerResponse> updateCaption(ServerRequest request) {
        String postId = request.pathVariable("postId");
        String userId = extractUserId(request);
        String username = extractUsername(request);

        log.info("Updating caption for post: {}, userId: {}, username: {}", postId, userId, username);

        return request.bodyToMono(Map.class)
                .flatMap(body -> {
                    String newCaption = (String) body.get("caption");
                    if (newCaption == null) {
                        return ServerResponse.badRequest()
                                .bodyValue(Map.of("error", "BAD_REQUEST", "message", "Caption es requerido"));
                    }
                    return postService.updateCaption(postId, newCaption, userId, username)
                            .then(ServerResponse.ok()
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .bodyValue(Map.of("message", "Caption actualizado exitosamente")));
                })
                .onErrorResume(this::handleError);
    }

    /**
     * Extrae el userId del header X-User-Id (inyectado por el gateway)
     */
    private String extractUserId(ServerRequest request) {
        return request.headers()
                .firstHeader("X-User-Id");
    }

    /**
     * Extrae el username del header X-Username (inyectado por el gateway)
     */
    private String extractUsername(ServerRequest request) {
        return request.headers()
                .firstHeader("X-Username");
    }

    /**
     * Manejo centralizado de errores
     */
    private Mono<ServerResponse> handleError(Throwable error) {
        log.error("Error in PostHandler: ", error);
        if (error instanceof ec.ups.upsglam.post.domain.exception.UnauthorizedException) {
            return ServerResponse.status(403)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(Map.of(
                            "error", "FORBIDDEN",
                            "message", error.getMessage()
                    ));
        }
        return ServerResponse.status(500)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "error", "INTERNAL_SERVER_ERROR",
                        "message", error.getMessage()
                ));
    }
}
