package ec.ups.upsglam.post.application.service;

import ec.ups.upsglam.post.domain.exception.PostNotFoundException;
import ec.ups.upsglam.post.domain.like.dto.LikeResponse;
import ec.ups.upsglam.post.infrastructure.firestore.document.LikeDocument;
import ec.ups.upsglam.post.infrastructure.firestore.repository.LikeFirestoreRepository;
import ec.ups.upsglam.post.infrastructure.firestore.repository.PostFirestoreRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;

/**
 * Servicio de dominio para Likes
 * Usa Firestore para gestionar likes como subcollections
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class LikeService {

    private final LikeFirestoreRepository likeRepository;
    private final PostFirestoreRepository postRepository;
    private final NotificationPublisher notificationPublisher;

    /**
     * Dar like a un post
     */
    public Mono<LikeResponse> likePost(String postId, String userId, String username) {
        // Verificar que el post existe
        return postRepository.findById(postId)
                .switchIfEmpty(Mono.error(new PostNotFoundException(postId)))
                .flatMap(post ->
                    // Verificar si ya dio like
                    likeRepository.existsByUserIdAndPostId(postId, userId)
                        .flatMap(exists -> {
                            if (exists) {
                                // Ya dio like, retornar estado actual
                                return buildLikeResponse(postId, userId, true);
                            }
                            // Crear nuevo like en Firestore
                            LikeDocument like = LikeDocument.builder()
                                    .userId(userId)
                                    .build();
                            
                            return likeRepository.addLike(postId, userId)
                                    .then(postRepository.updateLikesCount(postId, 1))
                                    .then(buildLikeResponse(postId, userId, true))
                                    .flatMap(res ->
                                            notificationPublisher.notifyLike(
                                                    post.getUserId(),
                                                    userId,
                                                    username,
                                                    postId)
                                                    .onErrorResume(e -> {
                                                        log.error("No se pudo crear notificación de like", e);
                                                        return Mono.empty();
                                                    })
                                                    .thenReturn(res)
                                    );
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
                    likeRepository.removeLike(postId, userId)
                            .then(postRepository.updateLikesCount(postId, -1))
                            .then(buildLikeResponse(postId, userId, false))
                )
                .doOnSuccess(response -> log.info("Usuario {} quitó like al post {}", userId, postId));
    }

    /**
     * Verificar si un usuario dio like a un post
     */
    public Mono<Boolean> hasUserLikedPost(String postId, String userId) {
        return likeRepository.existsByUserIdAndPostId(postId, userId);
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
