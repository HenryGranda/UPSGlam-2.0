package ec.ups.upsglam.post.infrastructure.repository;

import ec.ups.upsglam.post.domain.like.model.Like;
import org.springframework.data.r2dbc.repository.Query;
import org.springframework.data.repository.reactive.ReactiveCrudRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/**
 * Repositorio R2DBC para Likes
 * Accede a la tabla 'post_likes' en Supabase Postgres
 * Nota: Like tiene clave compuesta (post_id, user_id), no usa @Id
 */
@Repository
public interface LikeRepository extends ReactiveCrudRepository<Like, String> {

    /**
     * Verificar si un usuario ya dio like a un post
     */
    @Query("SELECT EXISTS(SELECT 1 FROM post_likes WHERE post_id = :postId AND user_id = :userId)")
    Mono<Boolean> existsByPostIdAndUserId(@Param("postId") String postId, @Param("userId") String userId);

    /**
     * Contar likes de un post
     */
    @Query("SELECT COUNT(*) FROM post_likes WHERE post_id = :postId")
    Mono<Long> countByPostId(@Param("postId") String postId);

    /**
     * Obtener usuarios que dieron like a un post
     */
    @Query("SELECT * FROM post_likes WHERE post_id = :postId ORDER BY created_at DESC")
    Flux<Like> findByPostId(@Param("postId") String postId);

    /**
     * Eliminar like de un usuario en un post
     */
    @Query("DELETE FROM post_likes WHERE post_id = :postId AND user_id = :userId")
    Mono<Void> deleteByPostIdAndUserId(@Param("postId") String postId, @Param("userId") String userId);

    /**
     * Obtener posts que le gustaron a un usuario
     */
    @Query("SELECT * FROM post_likes WHERE user_id = :userId ORDER BY created_at DESC")
    Flux<Like> findByUserId(@Param("userId") String userId);
}
