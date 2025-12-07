package ec.ups.upsglam.auth.domain.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Modelo de dominio para Usuario
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class User {
    private String id;           // Firebase UID
    private String email;
    private String username;
    private String fullName;
    private String photoUrl;
    private String bio;
    private Long createdAt;      // Timestamp
}
