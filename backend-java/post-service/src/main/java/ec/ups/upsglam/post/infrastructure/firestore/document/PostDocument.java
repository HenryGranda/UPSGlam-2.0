package ec.ups.upsglam.post.infrastructure.firestore.document;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * Documento de Post para Firestore
 * Colección: posts/{postId}
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PostDocument {

    private String id;
    private String userId;
    private String username;
    private String userPhotoUrl;
    private String imageUrl;
    private String filter;
    private String description;
    private Instant createdAt;
    private Integer likesCount;
    private Integer commentsCount;

    /**
     * Convierte el documento a Map para guardar en Firestore
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        if (userId != null) map.put("userId", userId);
        if (username != null) map.put("username", username);
        if (userPhotoUrl != null) map.put("userPhotoUrl", userPhotoUrl);
        if (imageUrl != null) map.put("imageUrl", imageUrl);
        if (filter != null) map.put("filter", filter);
        if (description != null) map.put("description", description);
        if (createdAt != null) map.put("createdAt", createdAt);
        if (likesCount != null) map.put("likesCount", likesCount);
        if (commentsCount != null) map.put("commentsCount", commentsCount);
        return map;
    }

    /**
     * Crea un documento desde un Map de Firestore
     */
    public static PostDocument fromMap(String id, Map<String, Object> data) {
        return PostDocument.builder()
                .id(id)
                .userId((String) data.get("userId"))
                .username((String) data.get("username"))
                .userPhotoUrl((String) data.get("userPhotoUrl"))
                .imageUrl((String) data.get("imageUrl"))
                .filter((String) data.get("filter"))
                .description((String) data.get("description"))
                .createdAt(convertToInstant(data.get("createdAt")))
                .likesCount(data.get("likesCount") != null ? 
                    ((Long) data.get("likesCount")).intValue() : 0)
                .commentsCount(data.get("commentsCount") != null ? 
                    ((Long) data.get("commentsCount")).intValue() : 0)
                .build();
    }

    /**
     * Convierte un objeto de Firestore a Instant
     */
    private static Instant convertToInstant(Object timestamp) {
        if (timestamp == null) {
            return null;
        }
        if (timestamp instanceof com.google.cloud.Timestamp) {
            return ((com.google.cloud.Timestamp) timestamp).toDate().toInstant();
        }
        if (timestamp instanceof Instant) {
            return (Instant) timestamp;
        }
        return null;
    }
}
