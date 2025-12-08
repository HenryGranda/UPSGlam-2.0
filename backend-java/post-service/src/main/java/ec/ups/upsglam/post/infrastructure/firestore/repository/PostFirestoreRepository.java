package ec.ups.upsglam.post.infrastructure.firestore.repository;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.*;
import ec.ups.upsglam.post.infrastructure.firestore.document.PostDocument;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.Map;
import java.util.UUID;

/**
 * Repositorio para la colección posts en Firestore
 * Colección: posts/{postId}
 */
@Repository
@Slf4j
public class PostFirestoreRepository {

    private final Firestore firestore;
    private static final String COLLECTION_NAME = "posts";

    public PostFirestoreRepository(Firestore firestore) {
        this.firestore = firestore;
    }

    /**
     * Guarda o actualiza un post en Firestore
     */
    public Mono<PostDocument> save(PostDocument post) {
        return Mono.fromCallable(() -> {
            // Si no tiene ID, generar uno nuevo
            if (post.getId() == null || post.getId().isEmpty()) {
                post.setId(UUID.randomUUID().toString());
            }
            
            // Si no tiene timestamp, usar ahora
            if (post.getCreatedAt() == null) {
                post.setCreatedAt(Instant.now());
            }
            
            // Inicializar contadores si son null
            if (post.getLikesCount() == null) {
                post.setLikesCount(0);
            }
            if (post.getCommentsCount() == null) {
                post.setCommentsCount(0);
            }

            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document(post.getId());
            ApiFuture<WriteResult> future = docRef.set(post.toMap());
            
            future.get(); // Wait for completion
            log.debug("Post guardado en Firestore: {}", post.getId());
            return post;
        });
    }

    /**
     * Busca un post por ID
     */
    public Mono<PostDocument> findById(String postId) {
        return Mono.fromCallable(() -> {
            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document(postId);
            ApiFuture<DocumentSnapshot> future = docRef.get();
            DocumentSnapshot document = future.get();

            if (document.exists()) {
                Map<String, Object> data = document.getData();
                return PostDocument.fromMap(document.getId(), data);
            }
            return null;
        });
    }

    /**
     * Obtiene el feed de posts (paginado y ordenado por fecha descendente)
     */
    public Flux<PostDocument> findFeed(int page, int size) {
        return Flux.defer(() -> {
            try {
                Query query = firestore.collection(COLLECTION_NAME)
                        .orderBy("createdAt", Query.Direction.DESCENDING)
                        .limit(size)
                        .offset(page * size);

                ApiFuture<QuerySnapshot> future = query.get();
                QuerySnapshot querySnapshot = future.get();

                return Flux.fromIterable(querySnapshot.getDocuments())
                        .map(doc -> PostDocument.fromMap(doc.getId(), doc.getData()));
            } catch (Exception e) {
                log.error("Error obteniendo feed desde Firestore", e);
                return Flux.error(new RuntimeException("Error obteniendo feed", e));
            }
        });
    }

    /**
     * Cuenta el total de posts
     */
    public Mono<Long> count() {
        return Mono.fromCallable(() -> {
            ApiFuture<QuerySnapshot> future = firestore.collection(COLLECTION_NAME).get();
            QuerySnapshot querySnapshot = future.get();
            return (long) querySnapshot.size();
        });
    }

    /**
     * Elimina un post por ID
     */
    public Mono<Void> delete(String postId) {
        return Mono.fromRunnable(() -> {
            try {
                DocumentReference docRef = firestore.collection(COLLECTION_NAME).document(postId);
                ApiFuture<WriteResult> future = docRef.delete();
                future.get();
                log.debug("Post eliminado de Firestore: {}", postId);
            } catch (Exception e) {
                log.error("Error eliminando post de Firestore: {}", postId, e);
                throw new RuntimeException("Error eliminando post", e);
            }
        });
    }

    /**
     * Actualiza el contador de likes de un post
     */
    public Mono<Void> updateLikesCount(String postId, int delta) {
        return Mono.fromRunnable(() -> {
            try {
                DocumentReference docRef = firestore.collection(COLLECTION_NAME).document(postId);
                ApiFuture<WriteResult> future = docRef.update("likesCount", FieldValue.increment(delta));
                future.get();
                log.debug("Likes count actualizado para post {}: delta={}", postId, delta);
            } catch (Exception e) {
                log.error("Error actualizando likesCount en Firestore: {}", postId, e);
                throw new RuntimeException("Error actualizando likesCount", e);
            }
        });
    }

    /**
     * Actualiza el contador de comentarios de un post
     */
    public Mono<Void> updateCommentsCount(String postId, int delta) {
        return Mono.fromRunnable(() -> {
            try {
                DocumentReference docRef = firestore.collection(COLLECTION_NAME).document(postId);
                ApiFuture<WriteResult> future = docRef.update("commentsCount", FieldValue.increment(delta));
                future.get();
                log.debug("Comments count actualizado para post {}: delta={}", postId, delta);
            } catch (Exception e) {
                log.error("Error actualizando commentsCount en Firestore: {}", postId, e);
                throw new RuntimeException("Error actualizando commentsCount", e);
            }
        });
    }
}
