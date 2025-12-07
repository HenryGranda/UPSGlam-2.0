package ec.ups.upsglam.post.application.service;

import ec.ups.upsglam.post.domain.exception.PostNotFoundException;
import ec.ups.upsglam.post.domain.exception.UnauthorizedException;
import ec.ups.upsglam.post.domain.post.dto.CreatePostRequest;
import ec.ups.upsglam.post.domain.post.dto.FeedResponse;
import ec.ups.upsglam.post.domain.post.dto.PostResponse;
import ec.ups.upsglam.post.domain.post.model.Post;
import ec.ups.upsglam.post.infrastructure.repository.LikeRepository;
import ec.ups.upsglam.post.infrastructure.repository.PostRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;

/**
 * Servicio de dominio para Posts
 * Maneja la lógica de negocio relacionada con posts
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class PostService {

    private final PostRepository postRepository;
    private final LikeRepository likeRepository;
    // private final ImageService imageService; // Se agregará después

    /**
     * Crear un nuevo post
     */
    public Mono<PostResponse> createPost(CreatePostRequest request, String userId) {
        log.info("Creando post para usuario: {}", userId);

        Post post = Post.builder()
                .id(java.util.UUID.randomUUID().toString())
                .userId(userId)
                .imageUrl(request.getMediaUrl())
                .filter(request.getFilter())
                .description(request.getCaption())
                .likesCount(0)
                .createdAt(LocalDateTime.now())
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
     * Obtener posts de un usuario
     */
    public Flux<PostResponse> getUserPosts(String userId, String currentUserId) {
        return postRepository.findByUserId(userId)
                .flatMap(post -> buildPostResponse(post, currentUserId));
    }

    /**
     * Obtener feed del usuario (posts de usuarios seguidos)
     */
    public Mono<FeedResponse> getUserFeed(String userId, int limit) {
        return postRepository.findFeedByUserId(userId, limit)
                .flatMap(post -> buildPostResponse(post, userId))
                .collectList()
                .map(posts -> FeedResponse.builder()
                        .posts(posts)
                        .hasMore(posts.size() >= limit)
                        .build());
    }

    /**
     * Buscar posts por hashtag
     */
    public Flux<PostResponse> searchByHashtag(String hashtag, String currentUserId) {
        return postRepository.findByHashtag(hashtag)
                .flatMap(post -> buildPostResponse(post, currentUserId));
    }

    /**
     * Eliminar un post (solo el autor puede eliminarlo)
     */
    public Mono<Void> deletePost(String postId, String userId) {
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> {
                    if (!post.getUserId().equals(userId)) {
                        return Mono.error(new UnauthorizedException("No tienes permiso para eliminar este post"));
                    }
                    return postRepository.deleteById(postId);
                })
                .doOnSuccess(v -> log.info("Post {} eliminado por usuario {}", postId, userId));
    }

    /**
     * Actualizar caption de un post
     */
    public Mono<PostResponse> updateCaption(String postId, String newCaption, String userId) {
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> {
                    if (!post.getUserId().equals(userId)) {
                        return Mono.error(new UnauthorizedException("No tienes permiso para editar este post"));
                    }
                    post.setDescription(newCaption);
                    return postRepository.save(post);
                })
                .flatMap(updatedPost -> buildPostResponse(updatedPost, userId));
    }

    /**
     * Construir PostResponse con información de likes
     */
    private Mono<PostResponse> buildPostResponse(Post post, String currentUserId) {
        Mono<Long> likesCount = likeRepository.countByPostId(post.getId());
        Mono<Boolean> likedByMe = currentUserId != null 
                ? likeRepository.existsByPostIdAndUserId(post.getId(), currentUserId)
                : Mono.just(false);

        return Mono.zip(likesCount, likedByMe)
                .map(tuple -> PostResponse.builder()
                        .id(post.getId())
                        .userId(post.getUserId())
                        .username(post.getUsername())
                        .userPhotoUrl(post.getUserPhotoUrl())
                        .imageUrl(post.getImageUrl())
                        .filter(post.getFilter())
                        .description(post.getDescription())
                        .likesCount(tuple.getT1().intValue())
                        .likedByMe(tuple.getT2())
                        .createdAt(post.getCreatedAt())
                        .build());
    }
}
