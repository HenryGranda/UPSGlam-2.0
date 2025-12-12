package ec.ups.upsglam.auth.domain.dto;

import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO para actualizar perfil de usuario
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class UpdateProfileRequest {
    
    @Size(min = 3, max = 20, message = "El username debe tener entre 3-20 caracteres")
    private String username;
    
    @Size(min = 2, max = 80, message = "El nombre debe tener entre 2-80 caracteres")
    private String fullName;
    
    @Size(max = 150, message = "La bio no puede tener más de 150 caracteres")
    private String bio;

    @Size(max = 100, message = "La URL de la foto no puede tener más de 100 caracteres")
    private String photoUrl;
}
