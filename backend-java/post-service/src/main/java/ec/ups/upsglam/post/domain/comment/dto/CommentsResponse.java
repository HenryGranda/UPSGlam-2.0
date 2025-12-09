package ec.ups.upsglam.post.domain.comment.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * DTO para respuesta paginada de comentarios
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CommentsResponse {

    private String postId;
    private List<CommentResponse> comments;
    private Integer totalCount;
}
