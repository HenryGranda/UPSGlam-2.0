package ec.ups.upsglam.auth.domain.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO para solicitud de seguir/dejar de seguir
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FollowRequest {
    private String targetUserId;  // ID del usuario a seguir/dejar de seguir
}
