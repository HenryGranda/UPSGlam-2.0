package ec.ups.upsglam.post.application.service;

import ec.ups.upsglam.post.domain.comment.dto.CommentRequest;
import ec.ups.upsglam.post.domain.comment.dto.CommentResponse;
import ec.ups.upsglam.post.domain.comment.dto.CommentsResponse;
import ec.ups.upsglam.post.domain.exception.PostNotFoundException;
import ec.ups.upsglam.post.domain.exception.UnauthorizedException;
import ec.ups.upsglam.post.infrastructure.firestore.document.CommentDocument;
import ec.ups.upsglam.post.infrastructure.firestore.repository.CommentFirestoreRepository;
import ec.ups.upsglam.post.infrastructure.firestore.repository.PostFirestoreRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.ZoneId;

/**
 * Servicio de dominio para Comments
 * Usa Firestore para gestionar comentarios como subcollections
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class CommentService {

    private final CommentFirestoreRepository commentRepository;
    private final PostFirestoreRepository postRepository;

    /**
     * Crear un comentario en un post
     */
    public Mono<CommentResponse> createComment(String postId, CommentRequest request, String userId) {
        log.info("Creando comentario en post {} por usuario {}", postId, userId);

        // Verificar que el post existe
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> {
                    CommentDocument comment = CommentDocument.builder()
                            .userId(userId)
                            .username(request.getUsername() != null ? request.getUsername() : "unknown")
                            .userPhotoUrl(request.getUserPhotoUrl())
                            .text(request.getText())
                            .build();

                    return commentRepository.save(postId, comment)
                            .flatMap(savedComment -> 
                                postRepository.updateCommentsCount(postId, 1)
                                    .thenReturn(savedComment)
                            );
                })
                .map(this::toCommentResponse)
                .doOnSuccess(c -> log.info("Comentario creado: {}", c.getId()));
    }

    /**
     * Obtener comentarios de un post (paginados)
     */
    public Mono<CommentsResponse> getPostComments(String postId, int page, int size) {
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> 
                    commentRepository.findByPostId(postId, page, size)
                            .map(this::toCommentResponse)
                            .collectList()
                            .zipWith(commentRepository.countByPostId(postId))
                            .map(tuple -> CommentsResponse.builder()
                                    .postId(postId)
                                    .comments(tuple.getT1())
                                    .totalCount(tuple.getT2().intValue())
                                    .build())
                );
    }

    /**
     * Obtener comentarios de un post (sin paginación)
     */
    public Mono<CommentsResponse> getPostComments(String postId) {
        return getPostComments(postId, 0, 50);
    }

    /**
     * Eliminar un comentario (solo el autor puede eliminarlo)
     */
    public Mono<Void> deleteComment(String postId, String commentId, String userId, String username) {
        return commentRepository.findById(postId, commentId)
            .switchIfEmpty(Mono.error(new RuntimeException("Comentario no encontrado")))
            .flatMap(comment -> {
                boolean isOwner = false;

                if (comment.getUserId() != null && userId != null &&
                    comment.getUserId().equals(userId)) {
                    isOwner = true;
                }

                if (!isOwner &&
                    comment.getUsername() != null && username != null &&
                    comment.getUsername().equals(username)) {
                    isOwner = true;
                }

                if (!isOwner) {
                    return Mono.error(new UnauthorizedException("No tienes permiso para eliminar este comentario"));
                }

                return commentRepository.delete(postId, commentId)
                        .then(postRepository.updateCommentsCount(postId, -1));
            })
            .doOnSuccess(v -> log.info("Comentario {} eliminado del post {}", commentId, postId));
    }

    public Mono<CommentResponse> updateComment(
            String postId,
            String commentId,
            CommentRequest request,
            String userId
    ) {
        return commentRepository.findById(postId, commentId)
            .switchIfEmpty(Mono.error(new RuntimeException("Comentario no encontrado")))
            .flatMap(comment -> {
                if (comment.getUserId() == null || !comment.getUserId().equals(userId)) {
                    return Mono.error(new UnauthorizedException("No tienes permiso para editar este comentario"));
                }

                comment.setText(request.getText());
                // opcionalmente actualizar username / photo si las mandas
                if (request.getUsername() != null) {
                    comment.setUsername(request.getUsername());
                }
                if (request.getUserPhotoUrl() != null) {
                    comment.setUserPhotoUrl(request.getUserPhotoUrl());
                }

                return commentRepository.save(postId, comment);
            })
            .map(this::toCommentResponse);
    }

    /**
     * Convertir CommentDocument a CommentResponse
     */
    private CommentResponse toCommentResponse(CommentDocument comment) {
        return CommentResponse.builder()
                .id(comment.getId())
                .postId(comment.getPostId())
                .userId(comment.getUserId())
                .username(comment.getUsername())
                .userPhotoUrl(comment.getUserPhotoUrl())
                .text(comment.getText())
                .createdAt(comment.getCreatedAt() != null 
                    ? comment.getCreatedAt().atZone(ZoneId.systemDefault()).toLocalDateTime()
                    : null)
                .build();
    }
}
