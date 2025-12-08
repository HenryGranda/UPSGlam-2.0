package ec.ups.upsglam.post.infrastructure.firestore.repository;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.*;
import ec.ups.upsglam.post.infrastructure.firestore.document.CommentDocument;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.UUID;

/**
 * Repositorio para la subcolección de comments en Firestore
 * Subcolección: posts/{postId}/comments/{commentId}
 */
@Repository
@Slf4j
public class CommentFirestoreRepository {

    private final Firestore firestore;
    private static final String POSTS_COLLECTION = "posts";
    private static final String COMMENTS_COLLECTION = "comments";

    public CommentFirestoreRepository(Firestore firestore) {
        this.firestore = firestore;
    }

    /**
     * Guarda un comentario en un post
     */
    public Mono<CommentDocument> save(String postId, CommentDocument comment) {
        return Mono.fromCallable(() -> {
            // Si no tiene ID, generar uno nuevo
            if (comment.getId() == null || comment.getId().isEmpty()) {
                comment.setId(UUID.randomUUID().toString());
            }
            
            // Si no tiene timestamp, usar ahora
            if (comment.getCreatedAt() == null) {
                comment.setCreatedAt(Instant.now());
            }

            DocumentReference commentRef = firestore
                    .collection(POSTS_COLLECTION)
                    .document(postId)
                    .collection(COMMENTS_COLLECTION)
                    .document(comment.getId());

            ApiFuture<WriteResult> future = commentRef.set(comment.toMap());
            future.get();
            
            log.debug("Comentario guardado: post={}, comment={}", postId, comment.getId());
            return comment;
        });
    }

    /**
     * Obtiene los comentarios de un post (paginados)
     */
    public Flux<CommentDocument> findByPostId(String postId, int page, int size) {
        return Flux.defer(() -> {
            try {
                Query query = firestore
                        .collection(POSTS_COLLECTION)
                        .document(postId)
                        .collection(COMMENTS_COLLECTION)
                        .orderBy("createdAt", Query.Direction.ASCENDING)
                        .limit(size)
                        .offset(page * size);

                ApiFuture<QuerySnapshot> future = query.get();
                QuerySnapshot querySnapshot = future.get();

                return Flux.fromIterable(querySnapshot.getDocuments())
                        .map(doc -> CommentDocument.fromMap(doc.getId(), postId, doc.getData()));
            } catch (Exception e) {
                log.error("Error obteniendo comentarios desde Firestore", e);
                return Flux.error(new RuntimeException("Error obteniendo comentarios", e));
            }
        });
    }

    /**
     * Cuenta los comentarios de un post
     */
    public Mono<Long> countByPostId(String postId) {
        return Mono.fromCallable(() -> {
            CollectionReference commentsRef = firestore
                    .collection(POSTS_COLLECTION)
                    .document(postId)
                    .collection(COMMENTS_COLLECTION);

            ApiFuture<QuerySnapshot> future = commentsRef.get();
            QuerySnapshot querySnapshot = future.get();
            
            return (long) querySnapshot.size();
        });
    }

    /**
     * Elimina un comentario
     */
    public Mono<Void> delete(String postId, String commentId) {
        return Mono.fromRunnable(() -> {
            try {
                DocumentReference commentRef = firestore
                        .collection(POSTS_COLLECTION)
                        .document(postId)
                        .collection(COMMENTS_COLLECTION)
                        .document(commentId);

                ApiFuture<WriteResult> future = commentRef.delete();
                future.get();
                
                log.debug("Comentario eliminado: post={}, comment={}", postId, commentId);
            } catch (Exception e) {
                log.error("Error eliminando comentario de Firestore", e);
                throw new RuntimeException("Error eliminando comentario", e);
            }
        });
    }

    /**
     * Busca un comentario por ID
     */
    public Mono<CommentDocument> findById(String postId, String commentId) {
        return Mono.fromCallable(() -> {
            DocumentReference commentRef = firestore
                    .collection(POSTS_COLLECTION)
                    .document(postId)
                    .collection(COMMENTS_COLLECTION)
                    .document(commentId);

            ApiFuture<DocumentSnapshot> future = commentRef.get();
            DocumentSnapshot document = future.get();

            if (document.exists()) {
                return CommentDocument.fromMap(document.getId(), postId, document.getData());
            }
            return null;
        });
    }
}
