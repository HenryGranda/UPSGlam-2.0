package ec.ups.upsglam.post.domain.post.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * DTO de respuesta para Post
 * Incluye el campo calculado 'likedByMe'
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PostResponse {

    private String id;

    @JsonProperty("userId")
    private String userId;

    private String username;

    @JsonProperty("userPhotoUrl")
    private String userPhotoUrl;

    @JsonProperty("imageUrl")
    private String imageUrl;

    private String filter;

    private String description;

    @JsonProperty("createdAt")
    private LocalDateTime createdAt;

    @JsonProperty("likesCount")
    private Integer likesCount;

    @JsonProperty("commentsCount")
    private Integer commentsCount;

    @JsonProperty("likedByMe")
    private Boolean likedByMe;
}
