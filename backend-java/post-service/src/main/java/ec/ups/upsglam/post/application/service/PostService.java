package ec.ups.upsglam.post.application.service;

import ec.ups.upsglam.post.domain.exception.PostNotFoundException;
import ec.ups.upsglam.post.domain.exception.UnauthorizedException;
import ec.ups.upsglam.post.domain.post.dto.CreatePostRequest;
import ec.ups.upsglam.post.domain.post.dto.FeedResponse;
import ec.ups.upsglam.post.domain.post.dto.PostResponse;
import ec.ups.upsglam.post.infrastructure.firestore.document.PostDocument;
import ec.ups.upsglam.post.infrastructure.firestore.repository.PostFirestoreRepository;
import ec.ups.upsglam.post.infrastructure.firestore.repository.LikeFirestoreRepository;
import ec.ups.upsglam.post.infrastructure.supabase.SupabaseStorageClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.ZoneId;

/**
 * Servicio de dominio para Posts
 * Usa Firestore para metadata y Supabase Storage para imágenes
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class PostService {

    private final PostFirestoreRepository postRepository;
    private final LikeFirestoreRepository likeRepository;
    private final SupabaseStorageClient storageClient;

    /**
     * Crear un nuevo post (sin imagen por ahora, solo metadata)
     */
    public Mono<PostResponse> createPost(CreatePostRequest request, String userId) {
        log.info("Creando post para usuario: {}", userId);

        PostDocument post = PostDocument.builder()
                .userId(userId)
                .username(request.getUsername() != null ? request.getUsername() : "unknown")
                .userPhotoUrl(request.getUserPhotoUrl())
                .imageUrl(request.getMediaUrl())
                .filter(request.getFilter())
                .description(request.getCaption())
                .build();

        return postRepository.save(post)
                .flatMap(savedPost -> buildPostResponse(savedPost, userId))
                .doOnSuccess(p -> log.info("Post creado exitosamente: {}", p.getId()))
                .doOnError(e -> log.error("Error al crear post", e));
    }

    /**
     * Obtener un post por ID
     */
    public Mono<PostResponse> getPostById(String postId, String currentUserId) {
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> buildPostResponse(post, currentUserId));
    }

    /**
     * Obtener feed paginado
     */
    public Mono<FeedResponse> getUserFeed(String userId, int limit) {
        return getFeed(0, limit, userId);
    }

    public Mono<FeedResponse> getFeed(int page, int size, String currentUserId) {
        return postRepository.findFeed(page, size)
                .flatMap(post -> buildPostResponse(post, currentUserId))
                .collectList()
                .zipWith(postRepository.count())
                .map(tuple -> FeedResponse.builder()
                        .posts(tuple.getT1())
                        .page(page)
                        .size(size)
                        .totalItems(tuple.getT2())
                        .hasMore((page + 1) * size < tuple.getT2())
                        .build());
    }

    /**
     * Eliminar un post (solo el autor puede eliminarlo)
     */
    public Mono<Void> deletePost(String postId, String userId) {
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> {
                    if (post.getUserId() == null || !post.getUserId().equals(userId)) {
                        return Mono.error(new UnauthorizedException("No tienes permiso para eliminar este post"));
                    }
                    
                    // Eliminar imagen de Supabase si existe
                    Mono<Void> deleteImage = Mono.just(post.getImageUrl())
                            .filter(url -> url != null && !url.isEmpty())
                            .map(storageClient::extractPathFromUrl)
                            .filter(path -> path != null)
                            .flatMap(storageClient::deleteFile)
                            .onErrorResume(e -> {
                                log.warn("Error eliminando imagen de post {}: {}", postId, e.getMessage());
                                return Mono.empty();
                            });
                    
                    // Eliminar metadata de Firestore
                    return deleteImage.then(postRepository.delete(postId));
                })
                .doOnSuccess(v -> log.info("Post eliminado: {}", postId))
                .doOnError(e -> log.error("Error eliminando post: {}", postId, e));
    }

    /**
     * Actualiza el caption de un post
     */
    public Mono<Void> updateCaption(String postId, String newCaption, String userId) {
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> {
                    if (post.getUserId() == null || !post.getUserId().equals(userId)) {
                        return Mono.error(new UnauthorizedException("No tienes permiso para editar este post"));
                    }
                    return postRepository.updateCaption(postId, newCaption);
                })
                .doOnSuccess(v -> log.info("Caption actualizado para post: {}", postId))
                .doOnError(e -> log.error("Error actualizando caption: {}", postId, e));
    }

    /**
     * Construir respuesta de Post con información adicional
     */
    private Mono<PostResponse> buildPostResponse(PostDocument post, String currentUserId) {
        // Verificar si el usuario actual le dio like
        Mono<Boolean> likedByMe = currentUserId != null
                ? likeRepository.existsByUserIdAndPostId(post.getId(), currentUserId)
                : Mono.just(false);

        return likedByMe.map(liked -> PostResponse.builder()
                .id(post.getId())
                .userId(post.getUserId())
                .username(post.getUsername())
                .userPhotoUrl(post.getUserPhotoUrl())
                .imageUrl(post.getImageUrl())
                .filter(post.getFilter())
                .description(post.getDescription())
                .createdAt(post.getCreatedAt() != null 
                    ? post.getCreatedAt().atZone(ZoneId.systemDefault()).toLocalDateTime()
                    : null)
                .likesCount(post.getLikesCount() != null ? post.getLikesCount() : 0)
                .commentsCount(post.getCommentsCount() != null ? post.getCommentsCount() : 0)
                .likedByMe(liked)
                .build());
    }
}
