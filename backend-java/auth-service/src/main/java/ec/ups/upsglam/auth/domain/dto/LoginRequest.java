package ec.ups.upsglam.auth.domain.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * DTO para login
 * identifier puede ser: email, username o teléfono
 */
@Data
public class LoginRequest {
    
    @NotBlank(message = "El identificador es obligatorio")
    private String identifier;  // email, username o phone
    
    @NotBlank(message = "La contraseña es obligatoria")
    private String password;
}
