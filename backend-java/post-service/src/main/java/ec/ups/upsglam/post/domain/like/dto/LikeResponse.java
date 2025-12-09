package ec.ups.upsglam.post.domain.like.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO de respuesta para operaciones de like/unlike
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LikeResponse {

    @JsonProperty("postId")
    private String postId;
    
    @JsonProperty("userId")
    private String userId;

    @JsonProperty("likesCount")
    private Integer likesCount;

    @JsonProperty("liked")
    private Boolean liked;
    
    private java.time.LocalDateTime createdAt;
}
