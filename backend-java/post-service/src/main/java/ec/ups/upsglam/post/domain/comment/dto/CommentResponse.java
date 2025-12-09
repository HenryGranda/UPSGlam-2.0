package ec.ups.upsglam.post.domain.comment.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * DTO de respuesta para Comment
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CommentResponse {

    private String id;

    @JsonProperty("postId")
    private String postId;

    @JsonProperty("userId")
    private String userId;

    private String username;

    @JsonProperty("userPhotoUrl")
    private String userPhotoUrl;

    private String text;

    @JsonProperty("createdAt")
    private LocalDateTime createdAt;
}
