package ec.ups.upsglam.auth.domain.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class NotificationResponse {
    private String id;
    private String userId;
    private String actorId;
    private String actorUsername;
    private String actorPhotoUrl;
    private String type; // like | comment | follow
    private String postId;
    private String commentId;
    private Long createdAt;
    private Boolean read;
}
