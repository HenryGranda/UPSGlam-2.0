package ec.ups.upsglam.post.domain.media.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Respuesta al crear una imagen temporal con filtro
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TempImageResponse {
    private String tempImageId;
    private String imageUrl;
    private String filter;
}
