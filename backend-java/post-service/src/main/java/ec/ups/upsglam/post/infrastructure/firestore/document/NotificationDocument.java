package ec.ups.upsglam.post.infrastructure.firestore.document;

import lombok.Builder;
import lombok.Data;

import java.util.HashMap;
import java.util.Map;

@Data
@Builder
public class NotificationDocument {
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

    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        if (userId != null) map.put("userId", userId);
        if (actorId != null) map.put("actorId", actorId);
        if (actorUsername != null) map.put("actorUsername", actorUsername);
        if (actorPhotoUrl != null) map.put("actorPhotoUrl", actorPhotoUrl);
        if (type != null) map.put("type", type);
        if (postId != null) map.put("postId", postId);
        if (commentId != null) map.put("commentId", commentId);
        map.put("createdAt", createdAt != null ? createdAt : System.currentTimeMillis());
        map.put("read", read != null ? read : false);
        return map;
    }
}
