package ec.ups.upsglam.post.application.service;

import ec.ups.upsglam.post.domain.exception.PostNotFoundException;
import ec.ups.upsglam.post.domain.like.dto.LikeResponse;
import ec.ups.upsglam.post.domain.like.model.Like;
import ec.ups.upsglam.post.infrastructure.repository.LikeRepository;
import ec.ups.upsglam.post.infrastructure.repository.PostRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;

/**
 * Servicio de dominio para Likes
 * Maneja la lógica de negocio relacionada con likes
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class LikeService {

    private final LikeRepository likeRepository;
    private final PostRepository postRepository;

    /**
     * Dar like a un post
     */
    public Mono<LikeResponse> likePost(String postId, String userId) {
        // Verificar que el post existe
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> 
                    // Verificar si ya dio like
                    likeRepository.existsByPostIdAndUserId(postId, userId)
                        .flatMap(exists -> {
                            if (exists) {
                                // Ya dio like, retornar estado actual
                                return buildLikeResponse(postId, userId, true);
                            }
                            // Crear nuevo like
                            Like like = Like.builder()
                                    .postId(postId)
                                    .userId(userId)
                                    .createdAt(LocalDateTime.now())
                                    .build();
                            
                            return likeRepository.save(like)
                                    .then(buildLikeResponse(postId, userId, true));
                        })
                )
                .doOnSuccess(response -> log.info("Usuario {} dio like al post {}", userId, postId))
                .doOnError(e -> log.error("Error al dar like", e));
    }

    /**
     * Quitar like de un post
     */
    public Mono<LikeResponse> unlikePost(String postId, String userId) {
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post -> 
                    likeRepository.deleteByPostIdAndUserId(postId, userId)
                            .then(buildLikeResponse(postId, userId, false))
                )
                .doOnSuccess(response -> log.info("Usuario {} quitó like al post {}", userId, postId));
    }

    /**
     * Obtener usuarios que dieron like a un post
     */
    public Flux<LikeResponse> getPostLikes(String postId) {
        return likeRepository.findByPostId(postId)
                .map(like -> LikeResponse.builder()
                        .postId(like.getPostId())
                        .userId(like.getUserId())
                        .liked(true)
                        .likesCount(0) // Se calculará después
                        .createdAt(like.getCreatedAt())
                        .build());
    }

    /**
     * Verificar si un usuario dio like a un post
     */
    public Mono<Boolean> hasUserLikedPost(String postId, String userId) {
        return likeRepository.existsByPostIdAndUserId(postId, userId);
    }

    /**
     * Construir respuesta de like con conteo
     */
    private Mono<LikeResponse> buildLikeResponse(String postId, String userId, boolean liked) {
        return likeRepository.countByPostId(postId)
                .map(count -> LikeResponse.builder()
                        .postId(postId)
                        .userId(userId)
                        .liked(liked)
                        .likesCount(count.intValue())
                        .createdAt(LocalDateTime.now())
                        .build());
    }
}
