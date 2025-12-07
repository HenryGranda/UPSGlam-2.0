package ec.ups.upsglam.post.domain.post.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Request para crear post desde tempImageId (estrategia con preview)
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class CreatePostRequest {

    @NotBlank(message = "tempImageId es requerido")
    private String tempImageId;

    @NotBlank(message = "filter es requerido")
    private String filter;

    @Size(max = 500, message = "El caption no puede exceder 500 caracteres")
    private String caption;
    
    private String mediaUrl;
    private String mediaType;
}
