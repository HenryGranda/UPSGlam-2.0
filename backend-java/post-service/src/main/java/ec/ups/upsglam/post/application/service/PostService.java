package ec.ups.upsglam.post.application.service;

import ec.ups.upsglam.post.domain.exception.PostNotFoundException;
import ec.ups.upsglam.post.domain.exception.UnauthorizedException;
import ec.ups.upsglam.post.domain.media.service.ImageService;
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
    private final ImageService imageService;

    /**
     * Crear un nuevo post
     * FLUJO CON FILTRO:
     * 1. Usuario ya tiene tempImageId de POST /images/preview
     * 2. Mover imagen de temp/ a posts/ con postId
     * 3. Guardar post en Firestore con URL final
     * 
     * FLUJO SIN FILTRO:
     * 1. Usuario sube imagen directa con POST /images/upload
     * 2. Ya viene mediaUrl final de Supabase
     * 3. Solo guardar post en Firestore
     */
    public Mono<PostResponse> createPost(CreatePostRequest request, String userId) {
        log.info("Creando post para usuario: {}", userId);

        // Si hay tempImageId, mover imagen de temp/ a posts/
        Mono<String> finalImageUrlMono;
        if (request.getTempImageId() != null && !request.getTempImageId().isEmpty()) {
            log.info("Moving temp image {} to posts/", request.getTempImageId());
            finalImageUrlMono = imageService.moveTempToPost(request.getTempImageId(), "post-" + userId + "-" + System.currentTimeMillis());
        } else {
            // Ya viene URL final (sin filtro)
            log.info("Using direct upload URL: {}", request.getMediaUrl());
            finalImageUrlMono = Mono.just(request.getMediaUrl());
        }

        return finalImageUrlMono
                .flatMap(finalImageUrl -> {
                    PostDocument post = PostDocument.builder()
                            .userId(userId)
                            .username(request.getUsername() != null ? request.getUsername() : "unknown")
                            .userPhotoUrl(request.getUserPhotoUrl())
                            .imageUrl(finalImageUrl)
                            .filter(request.getFilter())
                            .audioFile(request.getAudioFile())
                            .description(request.getCaption())
                            .build();

                    return postRepository.save(post)
                            .flatMap(savedPost -> buildPostResponse(savedPost, userId));
                })
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
     * Obtener posts de un usuario (acepta userId o username)
     */
    public Mono<FeedResponse> getUserPosts(String userRef, int page, int size, String currentUserId) {
        return postRepository.findByUser(userRef, page, size)
                .flatMap(post -> buildPostResponse(post, currentUserId))
                .collectList()
                .zipWith(postRepository.countByUser(userRef))
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
    public Mono<Void> deletePost(String postId, String userId, String username) {
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> {
                    boolean matchesUserId = userId != null && post.getUserId() != null && post.getUserId().equals(userId);
                    String reqUsername = normalizeUsername(username);
                    String postUsername = normalizeUsername(post.getUsername());
                    boolean matchesUsername = reqUsername != null && postUsername != null && reqUsername.equals(postUsername);
                    boolean allow = matchesUserId || matchesUsername;

                    if (!allow) {
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
    public Mono<Void> updateCaption(String postId, String newCaption, String userId, String username) {
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> {
                    boolean matchesUserId = userId != null && post.getUserId() != null && post.getUserId().equals(userId);
                    String reqUsername = normalizeUsername(username);
                    String postUsername = normalizeUsername(post.getUsername());
                    boolean matchesUsername = reqUsername != null && postUsername != null && reqUsername.equals(postUsername);
                    boolean allow = matchesUserId || matchesUsername;
                    if (!allow) {
                        return Mono.error(new UnauthorizedException("No tienes permiso para editar este post"));
                    }
                    return postRepository.updateCaption(postId, newCaption);
                })
                .doOnSuccess(v -> log.info("Caption actualizado para post: {}", postId))
                .doOnError(e -> log.error("Error actualizando caption: {}", postId, e));
    }

    private String normalizeUsername(String raw) {
        if (raw == null) return null;
        String clean = raw.trim();
        if (clean.startsWith("@")) {
            clean = clean.substring(1);
        }
        return clean.isEmpty() ? null : clean;
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
                .audioFile(post.getAudioFile())
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
