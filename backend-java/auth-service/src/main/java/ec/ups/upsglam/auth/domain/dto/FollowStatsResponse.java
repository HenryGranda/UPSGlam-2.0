package ec.ups.upsglam.auth.domain.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * DTO para estadísticas de follows de un usuario
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FollowStatsResponse {
    private String userId;
    private Long followersCount;
    private Long followingCount;
    private Boolean isFollowing;  // Si el usuario actual lo sigue (null si es el mismo usuario)
    private List<UserResponse> followers;   // Lista de seguidores (opcional)
    private List<UserResponse> following;   // Lista de seguidos (opcional)
}
