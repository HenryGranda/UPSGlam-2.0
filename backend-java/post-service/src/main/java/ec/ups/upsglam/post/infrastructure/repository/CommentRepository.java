package ec.ups.upsglam.post.infrastructure.repository;

import ec.ups.upsglam.post.domain.comment.model.Comment;
import org.springframework.data.r2dbc.repository.Query;
import org.springframework.data.r2dbc.repository.R2dbcRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/**
 * Repositorio R2DBC para Comments
 * Accede a la tabla 'comments' en Supabase Postgres
 */
@Repository
public interface CommentRepository extends R2dbcRepository<Comment, String> {

    /**
     * Obtener comentarios de un post ordenados por fecha
     */
    @Query("SELECT * FROM comments WHERE post_id = :postId ORDER BY created_at ASC")
    Flux<Comment> findByPostId(@Param("postId") String postId);

    /**
     * Contar comentarios de un post
     */
    @Query("SELECT COUNT(*) FROM comments WHERE post_id = :postId")
    Mono<Long> countByPostId(@Param("postId") String postId);

    /**
     * Obtener comentarios de un usuario
     */
    @Query("SELECT * FROM comments WHERE user_id = :userId ORDER BY created_at DESC")
    Flux<Comment> findByUserId(@Param("userId") String userId);

    /**
     * Eliminar comentario (solo si el usuario es el autor)
     */
    @Query("DELETE FROM comments WHERE id = :commentId AND user_id = :userId")
    Mono<Void> deleteByIdAndUserId(@Param("commentId") String commentId, @Param("userId") String userId);
}
