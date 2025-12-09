package ec.ups.upsglam.post.domain.media.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Request para subir imagen a Supabase Storage
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class UploadImageRequest {

    @NotBlank(message = "imageData es requerido")
    private String imageData; // Base64 encoded image
    
    private String fileName; // Opcional, se genera automáticamente si no se proporciona
}
