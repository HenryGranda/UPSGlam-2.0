package ec.ups.upsglam.post.application.service;

import com.google.cloud.firestore.Firestore;
import ec.ups.upsglam.post.infrastructure.firestore.document.NotificationDocument;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class NotificationPublisher {

    private final Firestore firestore;
    private static final String COLLECTION = "notifications";

    public Mono<Void> notifyLike(String ownerUserId, String actorId, String actorUsername, String postId) {
        return createNotification(ownerUserId, actorId, actorUsername, null, "like", postId, null);
    }

    public Mono<Void> notifyComment(String ownerUserId, String actorId, String actorUsername, String postId, String commentId) {
        return createNotification(ownerUserId, actorId, actorUsername, null, "comment", postId, commentId);
    }

    private Mono<Void> createNotification(String userId, String actorId, String actorUsername, String actorPhotoUrl, String type, String postId, String commentId) {
        if (userId == null || actorId == null || userId.equals(actorId)) {
            return Mono.empty();
        }
        return Mono.fromRunnable(() -> {
            try {
                NotificationDocument doc = NotificationDocument.builder()
                        .id(UUID.randomUUID().toString())
                        .userId(userId)
                        .actorId(actorId)
                        .actorUsername(actorUsername)
                        .actorPhotoUrl(actorPhotoUrl)
                        .type(type)
                        .postId(postId)
                        .commentId(commentId)
                        .createdAt(System.currentTimeMillis())
                        .read(false)
                        .build();

                firestore.collection(COLLECTION)
                        .document(doc.getId())
                        .set(doc.toMap())
                        .get();
                log.info("Notificación {} creada para user={} actor={}", type, userId, actorId);
            } catch (Exception e) {
                log.error("Error creando notificación {}", type, e);
            }
        }).subscribeOn(Schedulers.boundedElastic()).then();
    }
}
