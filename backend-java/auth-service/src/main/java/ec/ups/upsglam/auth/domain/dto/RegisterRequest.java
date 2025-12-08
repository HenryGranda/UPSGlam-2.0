package ec.ups.upsglam.auth.domain.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * DTO para registro de usuario
 */
@Data
public class RegisterRequest {
    
    @NotBlank(message = "El email es obligatorio")
    @Email(message = "Email inválido")
    private String email;
    
    @NotBlank(message = "La contraseña es obligatoria")
    @Size(min = 6, message = "La contraseña debe tener al menos 6 caracteres")
    private String password;
    
    @NotBlank(message = "El nombre completo es obligatorio")
    @Size(min = 2, max = 80, message = "El nombre debe tener entre 2 y 80 caracteres")
    private String fullName;
    
    @NotBlank(message = "El username es obligatorio")
    @Pattern(regexp = "^[a-zA-Z0-9_]{3,20}$", 
             message = "El username debe tener entre 3-20 caracteres (solo letras, números y guion bajo)")
    private String username;
}
