package ec.ups.upsglam.post.domain.media.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Respuesta al subir una imagen
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UploadImageResponse {
    private String imageId;
    private String imageUrl;
}
