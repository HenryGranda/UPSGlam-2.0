package ec.ups.upsglam.post.domain.comment.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Request para crear un comentario
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class CommentRequest {

    @NotBlank(message = "El comentario no puede estar vacío")
    @Size(min = 1, max = 500, message = "El comentario debe tener entre 1 y 500 caracteres")
    private String text;

    private String username;
    
    private String userPhotoUrl;
}
