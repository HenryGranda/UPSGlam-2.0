package ec.ups.upsglam.post.api.handler;

import ec.ups.upsglam.post.domain.comment.dto.CommentRequest;
import ec.ups.upsglam.post.application.service.CommentService;
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
public class CommentHandler {

    private static final Logger log = LoggerFactory.getLogger(CommentHandler.class);
    private final CommentService commentService;

    /**
     * GET /posts/{postId}/comments?page=0&size=20
     * Obtener comentarios de un post
     */
    public Mono<ServerResponse> getPostComments(ServerRequest request) {
        String postId = request.pathVariable("postId");
        int page = Integer.parseInt(request.queryParam("page").orElse("0"));
        int size = Integer.parseInt(request.queryParam("size").orElse("20"));

        log.info("Getting comments for post: {}, page: {}, size: {}", postId, page, size);

        return commentService.getPostComments(postId)
                .flatMap(commentsResponse ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(commentsResponse)
                )
                .onErrorResume(this::handleError);
    }

    /**
     * POST /posts/{postId}/comments
     * Crear un comentario en un post
     */
    public Mono<ServerResponse> createComment(ServerRequest request) {
        String postId = request.pathVariable("postId");
        String userId = extractUserId(request);
        String username = extractUsername(request);

        log.info("User {} creating comment on post {}", userId, postId);

        return request.bodyToMono(CommentRequest.class)
                .flatMap(commentRequest -> 
                    commentService.createComment(postId, commentRequest, userId)
                )
                .flatMap(commentResponse ->
                        ServerResponse.status(201)
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(commentResponse)
                )
                .onErrorResume(this::handleError);
    }

    /**
     * DELETE /posts/{postId}/comments/{commentId}
     * Eliminar un comentario (solo el autor o dueño del post)
     */
    public Mono<ServerResponse> deleteComment(ServerRequest request) {
        String postId = request.pathVariable("postId");
        String commentId = request.pathVariable("commentId");
        String userId = extractUserId(request);

        log.info("User {} deleting comment {} from post {}", userId, commentId, postId);

        return commentService.deleteComment(postId, commentId, userId)
                .then(ServerResponse.noContent().build())
                .onErrorResume(this::handleError);
    }

    /**
     * GET /users/{userId}/comments?page=0&size=20
     * Obtener comentarios de un usuario específico
     */
    public Mono<ServerResponse> getUserComments(ServerRequest request) {
        String userId = request.pathVariable("userId");
        int page = Integer.parseInt(request.queryParam("page").orElse("0"));
        int size = Integer.parseInt(request.queryParam("size").orElse("20"));

        log.info("Getting comments for user: {}, page: {}, size: {}", userId, page, size);

        // Por ahora retornamos lista vacía, este método requiere implementación adicional
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "postId", "",
                        "comments", java.util.Collections.emptyList(),
                        "totalCount", 0
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
        log.error("Error in CommentHandler: ", error);
        
        int statusCode = 500;
        String errorCode = "INTERNAL_SERVER_ERROR";
        
        if (error instanceof IllegalArgumentException) {
            statusCode = 400;
            errorCode = "BAD_REQUEST";
        }
        
        return ServerResponse.status(statusCode)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "error", errorCode,
                        "message", error.getMessage()
                ));
    }
}
