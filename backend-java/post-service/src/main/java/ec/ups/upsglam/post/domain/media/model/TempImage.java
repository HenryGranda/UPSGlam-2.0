package ec.ups.upsglam.post.domain.media.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Representa una imagen temporal en Supabase Storage
 * Usada en el flujo de preview con CUDA
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TempImage {

    @JsonProperty("tempImageId")
    private String tempImageId;

    @JsonProperty("imageUrl")
    private String imageUrl;
}
