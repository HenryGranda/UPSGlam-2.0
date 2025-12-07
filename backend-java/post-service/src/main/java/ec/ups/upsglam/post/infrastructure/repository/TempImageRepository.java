package ec.ups.upsglam.post.infrastructure.repository;

import ec.ups.upsglam.post.domain.media.model.TempImage;
import org.springframework.data.r2dbc.repository.Query;
import org.springframework.data.r2dbc.repository.R2dbcRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;

/**
 * Repositorio R2DBC para TempImages
 * Accede a la tabla 'temp_images' en Supabase Postgres
 */
@Repository
public interface TempImageRepository extends R2dbcRepository<TempImage, String> {

    /**
     * Buscar imagen temporal por usuario
     */
    @Query("SELECT * FROM temp_images WHERE user_id = :userId ORDER BY created_at DESC")
    Flux<TempImage> findByUserId(@Param("userId") String userId);

    /**
     * Eliminar imágenes temporales expiradas (más de 1 hora)
     */
    @Query("DELETE FROM temp_images WHERE created_at < :expirationTime")
    Mono<Void> deleteExpiredImages(@Param("expirationTime") LocalDateTime expirationTime);

    /**
     * Eliminar imágenes temporales de un usuario
     */
    @Query("DELETE FROM temp_images WHERE user_id = :userId")
    Mono<Void> deleteByUserId(@Param("userId") String userId);
}
