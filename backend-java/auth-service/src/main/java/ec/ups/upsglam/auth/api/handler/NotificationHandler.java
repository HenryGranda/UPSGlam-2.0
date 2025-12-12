package ec.ups.upsglam.auth.api.handler;

import ec.ups.upsglam.auth.application.NotificationService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.Map;

@Component
@RequiredArgsConstructor
@Slf4j
public class NotificationHandler {

    private final NotificationService notificationService;

    private String extractUserId(ServerRequest request) {
        String tokenUserId = request.headers().firstHeader("X-User-Id");
        return tokenUserId;
    }

    public Mono<ServerResponse> list(ServerRequest request) {
        String userId = extractUserId(request);
        if (userId == null || userId.isBlank()) {
            return ServerResponse.status(401).bodyValue(Map.of(
                    "code", "UNAUTHORIZED",
                    "message", "Token requerido"
            ));
        }
        int limit = Integer.parseInt(request.queryParam("limit").orElse("20"));

        return notificationService.listNotifications(userId, limit)
                .flatMap(list -> ServerResponse.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(Map.of("items", list)))
                .onErrorResume(e -> {
                    log.error("Error listando notificaciones", e);
                    return ServerResponse.status(500).bodyValue(Map.of(
                            "code", "INTERNAL_ERROR",
                            "message", e.getMessage()
                    ));
                });
    }

    public Mono<ServerResponse> unreadCount(ServerRequest request) {
        String userId = extractUserId(request);
        if (userId == null || userId.isBlank()) {
            return ServerResponse.status(401).bodyValue(Map.of(
                    "code", "UNAUTHORIZED",
                    "message", "Token requerido"
            ));
        }
        return notificationService.unreadCount(userId)
                .flatMap(count -> ServerResponse.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(Map.of("unread", count)))
                .onErrorResume(e -> ServerResponse.status(500).bodyValue(Map.of(
                        "code", "INTERNAL_ERROR",
                        "message", e.getMessage()
                )));
    }

    public Mono<ServerResponse> markRead(ServerRequest request) {
        String userId = extractUserId(request);
        if (userId == null || userId.isBlank()) {
            return ServerResponse.status(401).bodyValue(Map.of(
                    "code", "UNAUTHORIZED",
                    "message", "Token requerido"
            ));
        }
        String id = request.pathVariable("id");
        return notificationService.markAsRead(userId, id)
                .then(ServerResponse.noContent().build())
                .onErrorResume(e -> {
                    log.error("Error marcando notificación como leída", e);
                    return ServerResponse.status(500).bodyValue(Map.of(
                            "code", "INTERNAL_ERROR",
                            "message", e.getMessage()
                    ));
                });
    }

    public Mono<ServerResponse> saveFcmToken(ServerRequest request) {
        String userId = extractUserId(request);
        if (userId == null || userId.isBlank()) {
            return ServerResponse.status(401).bodyValue(Map.of(
                    "code", "UNAUTHORIZED",
                    "message", "Token requerido"
            ));
        }
        return request.bodyToMono(Map.class)
                .flatMap(body -> {
                    String token = (String) body.get("fcmToken");
                    if (token == null || token.isBlank()) {
                        return ServerResponse.badRequest().bodyValue(Map.of(
                                "code", "INVALID_TOKEN",
                                "message", "fcmToken requerido"
                        ));
                    }
                    return notificationService.saveFcmToken(userId, token)
                            .then(ServerResponse.noContent().build());
                })
                .onErrorResume(e -> ServerResponse.status(500).bodyValue(Map.of(
                        "code", "INTERNAL_ERROR",
                        "message", e.getMessage()
                )));
    }
}
