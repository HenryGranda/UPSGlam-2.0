package ec.ups.upsglam.post.infrastructure.firestore.document;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * Documento de Like para Firestore
 * Subcolección: posts/{postId}/likes/{userId}
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LikeDocument {

    private String userId;
    private Instant createdAt;

    /**
     * Convierte el documento a Map para guardar en Firestore
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        if (userId != null) map.put("userId", userId);
        if (createdAt != null) map.put("createdAt", createdAt);
        return map;
    }

    /**
     * Crea un documento desde un Map de Firestore
     */
    public static LikeDocument fromMap(Map<String, Object> data) {
        return LikeDocument.builder()
                .userId((String) data.get("userId"))
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
