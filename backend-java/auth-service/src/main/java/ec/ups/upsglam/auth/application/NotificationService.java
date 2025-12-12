package ec.ups.upsglam.auth.application;

import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import ec.ups.upsglam.auth.domain.dto.NotificationResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class NotificationService {

    private static final String COLLECTION = "notifications";
    private final Firestore firestore;

    public Mono<Void> createFollowNotification(String userId, String actorId, String actorUsername, String actorPhotoUrl) {
        if (userId == null || actorId == null || userId.equals(actorId)) {
            return Mono.empty();
        }
        return Mono.fromRunnable(() -> {
            try {
                String id = java.util.UUID.randomUUID().toString();
                Map<String, Object> data = new HashMap<>();
                data.put("userId", userId);
                data.put("actorId", actorId);
                data.put("actorUsername", actorUsername);
                data.put("actorPhotoUrl", actorPhotoUrl);
                data.put("type", "follow");
                data.put("createdAt", System.currentTimeMillis());
                data.put("read", false);
                firestore.collection(COLLECTION).document(id).set(data).get();
                log.info("Notificación follow creada {} -> {}", actorId, userId);
            } catch (Exception e) {
                log.error("Error creando notificación follow", e);
                throw new RuntimeException(e);
            }
        }).subscribeOn(Schedulers.boundedElastic()).then();
    }

    public Mono<List<NotificationResponse>> listNotifications(String userId, int limit) {
        return Mono.fromCallable(() -> {
            var query = firestore.collection(COLLECTION)
                    .whereEqualTo("userId", userId)
                    .orderBy("createdAt", com.google.cloud.firestore.Query.Direction.DESCENDING)
                    .limit(limit);

            List<QueryDocumentSnapshot> docs = query.get().get().getDocuments();
            return docs.stream()
                    .map(this::toResponse)
                    .collect(Collectors.toList());
        }).subscribeOn(Schedulers.boundedElastic());
    }

    public Mono<Long> unreadCount(String userId) {
        return Mono.fromCallable(() -> {
            var query = firestore.collection(COLLECTION)
                    .whereEqualTo("userId", userId)
                    .whereEqualTo("read", false);
            return (long) query.get().get().size();
        }).subscribeOn(Schedulers.boundedElastic());
    }

    public Mono<Void> markAsRead(String userId, String notificationId) {
        return Mono.fromRunnable(() -> {
            try {
                var docRef = firestore.collection(COLLECTION).document(notificationId);
                var snapshot = docRef.get().get();
                if (!snapshot.exists()) return;
                var data = snapshot.getData();
                if (data == null) return;
                String owner = Objects.toString(data.get("userId"), "");
                if (!owner.equals(userId)) {
                    throw new RuntimeException("No autorizado a marcar esta notificación");
                }
                Map<String, Object> updates = new HashMap<>();
                updates.put("read", true);
                docRef.update(updates).get();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }).subscribeOn(Schedulers.boundedElastic()).then();
    }

    /**
     * Almacena/actualiza el token FCM del usuario en el documento users/{uid}.
     */
    public Mono<Void> saveFcmToken(String userId, String fcmToken) {
        return Mono.fromRunnable(() -> {
            try {
                Map<String, Object> updates = new HashMap<>();
                updates.put("fcmToken", fcmToken);
                firestore.collection("users").document(userId).update(updates).get();
                log.info("FCM token actualizado para {}", userId);
            } catch (Exception e) {
                log.error("Error guardando FCM token para {}", userId, e);
                throw new RuntimeException("No se pudo guardar FCM token");
            }
        }).subscribeOn(Schedulers.boundedElastic()).then();
    }

    private NotificationResponse toResponse(QueryDocumentSnapshot doc) {
        var data = doc.getData();
        return NotificationResponse.builder()
                .id(doc.getId())
                .userId((String) data.get("userId"))
                .actorId((String) data.get("actorId"))
                .actorUsername((String) data.get("actorUsername"))
                .actorPhotoUrl((String) data.get("actorPhotoUrl"))
                .type((String) data.get("type"))
                .postId((String) data.get("postId"))
                .commentId((String) data.get("commentId"))
                .createdAt((Long) data.get("createdAt"))
                .read((Boolean) data.getOrDefault("read", false))
                .build();
    }
}
