package ec.ups.upsglam.post.application.service;

import ec.ups.upsglam.post.domain.comment.dto.CommentRequest;
import ec.ups.upsglam.post.domain.comment.dto.CommentResponse;
import ec.ups.upsglam.post.domain.comment.dto.CommentsResponse;
import ec.ups.upsglam.post.domain.comment.model.Comment;
import ec.ups.upsglam.post.domain.exception.PostNotFoundException;
import ec.ups.upsglam.post.domain.exception.UnauthorizedException;
import ec.ups.upsglam.post.infrastructure.repository.CommentRepository;
import ec.ups.upsglam.post.infrastructure.repository.PostRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;

/**
 * Servicio de dominio para Comments
 * Maneja la lógica de negocio relacionada con comentarios
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class CommentService {

    private final CommentRepository commentRepository;
    private final PostRepository postRepository;

    /**
     * Crear un comentario en un post
     */
    public Mono<CommentResponse> createComment(String postId, CommentRequest request, String userId) {
        log.info("Creando comentario en post {} por usuario {}", postId, userId);

        // Verificar que el post existe
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> {
                    Comment comment = Comment.builder()
                            .id(java.util.UUID.randomUUID().toString())
                            .postId(postId)
                            .userId(userId)
                            .text(request.getText())
                            .createdAt(LocalDateTime.now())
                            .build();

                    return commentRepository.save(comment);
                })
                .map(this::toCommentResponse)
                .doOnSuccess(c -> log.info("Comentario creado: {}", c.getId()));
    }

    /**
     * Obtener comentarios de un post
     */
    public Mono<CommentsResponse> getPostComments(String postId) {
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> 
                    commentRepository.findByPostId(postId)
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
     * Eliminar un comentario (solo el autor puede eliminarlo)
     */
    public Mono<Void> deleteComment(String commentId, String userId) {
        return commentRepository.findById(commentId)
                .switchIfEmpty(Mono.error(new RuntimeException("Comentario no encontrado")))
                .flatMap(comment -> {
                    if (!comment.getUserId().equals(userId)) {
                        return Mono.error(new UnauthorizedException("No tienes permiso para eliminar este comentario"));
                    }
                    return commentRepository.deleteById(commentId);
                })
                .doOnSuccess(v -> log.info("Comentario {} eliminado", commentId));
    }

    /**
     * Obtener comentarios de un usuario
     */
    public Mono<CommentsResponse> getUserComments(String userId) {
        return commentRepository.findByUserId(userId)
                .map(this::toCommentResponse)
                .collectList()
                .map(comments -> CommentsResponse.builder()
                        .postId(null) // No es de un post específico
                        .comments(comments)
                        .totalCount(comments.size())
                        .build());
    }

    /**
     * Convertir Comment a CommentResponse
     */
    private CommentResponse toCommentResponse(Comment comment) {
        return CommentResponse.builder()
                .id(comment.getId())
                .postId(comment.getPostId())
                .userId(comment.getUserId())
                .text(comment.getText())
                .createdAt(comment.getCreatedAt())
                .build();
    }
}
