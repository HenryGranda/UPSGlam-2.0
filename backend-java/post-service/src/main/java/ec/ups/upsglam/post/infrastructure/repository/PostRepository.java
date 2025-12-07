package ec.ups.upsglam.post.infrastructure.repository;

import ec.ups.upsglam.post.domain.post.model.Post;
import org.springframework.data.r2dbc.repository.Query;
import org.springframework.data.r2dbc.repository.R2dbcRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;



/**
 * Repositorio R2DBC para Posts
 * Accede a la tabla 'posts' en Supabase Postgres
 */
@Repository
public interface PostRepository extends R2dbcRepository<Post, String> {

    /**
     * Obtener posts de un usuario específico ordenados por fecha descendente
     */
    @Query("SELECT * FROM posts WHERE user_id = :userId ORDER BY created_at DESC")
    Flux<Post> findByUserId(@Param("userId") String userId);

    /**
     * Obtener feed de posts de usuarios seguidos
     * (Para implementar cuando tengamos servicio de usuarios)
     */
    @Query("SELECT p.* FROM posts p " +
           "INNER JOIN follows f ON p.user_id = f.following_id " +
           "WHERE f.follower_id = :userId " +
           "ORDER BY p.created_at DESC " +
           "LIMIT :limit")
    Flux<Post> findFeedByUserId(@Param("userId") String userId, @Param("limit") int limit);

    /**
     * Buscar posts por hashtags (búsqueda por coincidencia parcial)
     */
    @Query("SELECT * FROM posts WHERE caption ILIKE CONCAT('%', :hashtag, '%') ORDER BY created_at DESC")
    Flux<Post> findByHashtag(@Param("hashtag") String hashtag);

    /**
     * Contar posts de un usuario
     */
    @Query("SELECT COUNT(*) FROM posts WHERE user_id = :userId")
    Mono<Long> countByUserId(@Param("userId") String userId);
}
