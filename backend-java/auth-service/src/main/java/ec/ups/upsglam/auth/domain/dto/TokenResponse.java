package ec.ups.upsglam.auth.domain.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO para token de Firebase
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TokenResponse {
    private String idToken;
    private String refreshToken;
    private Long expiresIn;  // segundos
}
