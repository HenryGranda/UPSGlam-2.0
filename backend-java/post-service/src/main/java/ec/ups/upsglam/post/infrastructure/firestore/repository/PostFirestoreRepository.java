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
import java.util.Objects;

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
     * Obtiene posts de un usuario por userId o username
     */
    public Flux<PostDocument> findByUser(String userRef, int page, int size) {
        return Flux.defer(() -> {
            try {
                // Cargamos todos los posts y filtramos en memoria (dataset reducido)
                QuerySnapshot allSnapshot = firestore.collection(COLLECTION_NAME)
                        .get()
                        .get();

                String refLower = userRef == null ? "" : userRef.trim().toLowerCase();

                var documents = allSnapshot.getDocuments().stream()
                        .filter(doc -> {
                            String userId = doc.getString("userId");
                            String username = doc.getString("username");

                            boolean matchId = userId != null && userId.equals(refLower);
                            boolean matchUsername = username != null && username.equalsIgnoreCase(refLower);

                            // aceptar coincidencias de username con o sin prefijo @
                            if (!matchUsername && username != null && username.startsWith("@")) {
                                matchUsername = username.substring(1).equalsIgnoreCase(refLower);
                            }

                            return matchId || matchUsername;
                        })
                        .sorted((a, b) -> {
                            Instant ta = safeInstant(a.get("createdAt"));
                            Instant tb = safeInstant(b.get("createdAt"));
                            if (ta == null && tb == null) return 0;
                            if (ta == null) return 1;
                            if (tb == null) return -1;
                            return tb.compareTo(ta);
                        })
                        .skip((long) page * size)
                        .limit(size)
                        .toList();

                return Flux.fromIterable(documents)
                        .map(doc -> PostDocument.fromMap(doc.getId(), doc.getData()));
            } catch (Exception e) {
                log.error("Error obteniendo posts de usuario desde Firestore", e);
                return Flux.error(new RuntimeException("Error obteniendo posts de usuario", e));
            }
        });
    }

    /**
     * Cuenta posts de un usuario por userId o username
     */
    public Mono<Long> countByUser(String userRef) {
        return Mono.fromCallable(() -> {
            QuerySnapshot allSnapshot = firestore.collection(COLLECTION_NAME)
                    .get()
                    .get();

            String refLower = userRef == null ? "" : userRef.trim().toLowerCase();

            return allSnapshot.getDocuments().stream()
                    .filter(doc -> {
                        String userId = doc.getString("userId");
                        String username = doc.getString("username");

                        boolean matchId = userId != null && userId.equals(refLower);
                        boolean matchUsername = username != null && username.equalsIgnoreCase(refLower);

                        if (!matchUsername && username != null && username.startsWith("@")) {
                            matchUsername = username.substring(1).equalsIgnoreCase(refLower);
                        }

                        return matchId || matchUsername;
                    })
                    .count();
        });
    }

    /**
     * Convierte el campo createdAt (que puede venir como Timestamp, Instant, Map o numero) a Instant
     */
    private Instant safeInstant(Object raw) {
        try {
            if (raw == null) return null;
            if (raw instanceof com.google.cloud.Timestamp ts) {
                return ts.toDate().toInstant();
            }
            if (raw instanceof Instant instant) {
                return instant;
            }
            if (raw instanceof Number num) {
                // epoch millis
                return Instant.ofEpochMilli(num.longValue());
            }
            if (raw instanceof Map<?, ?> map) {
                Object secObj = map.get("seconds");
                if (secObj == null) secObj = map.get("_seconds");
                Object nanoObj = map.get("nanos");
                if (nanoObj == null) nanoObj = map.get("_nanoseconds");

                long seconds = secObj instanceof Number ? ((Number) secObj).longValue() : 0L;
                long nanos = nanoObj instanceof Number ? ((Number) nanoObj).longValue() : 0L;
                return Instant.ofEpochSecond(seconds, nanos);
            }
        } catch (Exception ignored) {
        }
        return null;
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

    /**
     * Actualiza el caption de un post
     */
    public Mono<Void> updateCaption(String postId, String newCaption) {
        return Mono.fromRunnable(() -> {
            try {
                DocumentReference docRef = firestore.collection(COLLECTION_NAME).document(postId);
                ApiFuture<WriteResult> future = docRef.update("caption", newCaption);
                future.get();
                log.debug("Caption actualizado para post {}: {}", postId, newCaption);
            } catch (Exception e) {
                log.error("Error actualizando caption en Firestore: {}", postId, e);
                throw new RuntimeException("Error actualizando caption", e);
            }
        });
    }
}
