package ec.ups.upsglam.auth.domain.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Modelo de dominio para Follow (relación de seguimiento)
 * Representa que followerUserId sigue a followedUserId
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Follow {
    private String id;                  // ID único del documento
    private String followerUserId;      // ID del usuario que sigue
    private String followedUserId;      // ID del usuario seguido
    private Long createdAt;            // Timestamp de cuando se creó el follow
}
