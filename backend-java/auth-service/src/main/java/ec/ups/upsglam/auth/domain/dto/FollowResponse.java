package ec.ups.upsglam.auth.domain.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO de respuesta para operaciones de follow
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FollowResponse {
    private boolean success;
    private String message;
    private boolean isFollowing;  // Estado actual después de la operación
    private Long followersCount;  // Nuevo conteo de seguidores del usuario seguido
}
