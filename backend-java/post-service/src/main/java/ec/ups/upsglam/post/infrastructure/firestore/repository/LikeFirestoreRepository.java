package ec.ups.upsglam.post.infrastructure.firestore.repository;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.*;
import ec.ups.upsglam.post.infrastructure.firestore.document.LikeDocument;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Mono;

import java.time.Instant;

/**
 * Repositorio para la subcolección de likes en Firestore
 * Subcolección: posts/{postId}/likes/{userId}
 */
@Repository
@Slf4j
public class LikeFirestoreRepository {

    private final Firestore firestore;
    private static final String POSTS_COLLECTION = "posts";
    private static final String LIKES_COLLECTION = "likes";

    public LikeFirestoreRepository(Firestore firestore) {
        this.firestore = firestore;
    }

    /**
     * Agrega un like a un post
     */
    public Mono<LikeDocument> addLike(String postId, String userId) {
        return Mono.fromCallable(() -> {
            LikeDocument like = LikeDocument.builder()
                    .userId(userId)
                    .createdAt(Instant.now())
                    .build();

            DocumentReference likeRef = firestore
                    .collection(POSTS_COLLECTION)
                    .document(postId)
                    .collection(LIKES_COLLECTION)
                    .document(userId);

            ApiFuture<WriteResult> future = likeRef.set(like.toMap());
            future.get();
            
            log.debug("Like agregado: post={}, user={}", postId, userId);
            return like;
        });
    }

    /**
     * Elimina un like de un post
     */
    public Mono<Void> removeLike(String postId, String userId) {
        return Mono.fromRunnable(() -> {
            try {
                DocumentReference likeRef = firestore
                        .collection(POSTS_COLLECTION)
                        .document(postId)
                        .collection(LIKES_COLLECTION)
                        .document(userId);

                ApiFuture<WriteResult> future = likeRef.delete();
                future.get();
                
                log.debug("Like eliminado: post={}, user={}", postId, userId);
            } catch (Exception e) {
                log.error("Error eliminando like en Firestore", e);
                throw new RuntimeException("Error eliminando like", e);
            }
        });
    }

    /**
     * Verifica si un usuario ya dio like a un post
     */
    public Mono<Boolean> existsByUserIdAndPostId(String postId, String userId) {
        return Mono.fromCallable(() -> {
            DocumentReference likeRef = firestore
                    .collection(POSTS_COLLECTION)
                    .document(postId)
                    .collection(LIKES_COLLECTION)
                    .document(userId);

            ApiFuture<DocumentSnapshot> future = likeRef.get();
            DocumentSnapshot document = future.get();
            
            return document.exists();
        });
    }

    /**
     * Cuenta los likes de un post
     */
    public Mono<Long> countByPostId(String postId) {
        return Mono.fromCallable(() -> {
            CollectionReference likesRef = firestore
                    .collection(POSTS_COLLECTION)
                    .document(postId)
                    .collection(LIKES_COLLECTION);

            ApiFuture<QuerySnapshot> future = likesRef.get();
            QuerySnapshot querySnapshot = future.get();
            
            return (long) querySnapshot.size();
        });
    }
}
