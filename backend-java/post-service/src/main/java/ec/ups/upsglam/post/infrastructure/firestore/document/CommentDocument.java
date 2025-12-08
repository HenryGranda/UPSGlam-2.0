package ec.ups.upsglam.post.infrastructure.firestore.document;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * Documento de Comment para Firestore
 * Subcolección: posts/{postId}/comments/{commentId}
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CommentDocument {

    private String id;
    private String postId;
    private String userId;
    private String username;
    private String userPhotoUrl;
    private String text;
    private Instant createdAt;

    /**
     * Convierte el documento a Map para guardar en Firestore
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        if (userId != null) map.put("userId", userId);
        if (username != null) map.put("username", username);
        if (userPhotoUrl != null) map.put("userPhotoUrl", userPhotoUrl);
        if (text != null) map.put("text", text);
        if (createdAt != null) map.put("createdAt", createdAt);
        return map;
    }

    /**
     * Crea un documento desde un Map de Firestore
     */
    public static CommentDocument fromMap(String id, String postId, Map<String, Object> data) {
        return CommentDocument.builder()
                .id(id)
                .postId(postId)
                .userId((String) data.get("userId"))
                .username((String) data.get("username"))
                .userPhotoUrl((String) data.get("userPhotoUrl"))
                .text((String) data.get("text"))
                .createdAt(convertToInstant(data.get("createdAt")))
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
