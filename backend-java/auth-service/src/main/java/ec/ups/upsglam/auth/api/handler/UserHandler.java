package ec.ups.upsglam.auth.api.handler;

import ec.ups.upsglam.auth.application.UserService;
import ec.ups.upsglam.auth.domain.dto.UpdateProfileRequest;
import ec.ups.upsglam.auth.domain.exception.InvalidCredentialsException;
import ec.ups.upsglam.auth.domain.exception.UserNotFoundException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.Map;
import java.util.Optional;

import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

/**
 * Handler para endpoints de usuario
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UserHandler {

    private final UserService userService;

    /**
     * PATCH /users/me
     */
    public Mono<ServerResponse> updateProfile(ServerRequest request) {
        String authHeader = request.headers().firstHeader("Authorization");

        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return ServerResponse.status(401)
                    .bodyValue(java.util.Map.of(
                            "code", "UNAUTHORIZED",
                            "message", "Token requerido"
                    ));
        }

        String token = authHeader.substring(7);

        return request.bodyToMono(UpdateProfileRequest.class)
                .flatMap(updateRequest -> userService.updateProfile(token, updateRequest))
                .flatMap(user -> ServerResponse.ok().bodyValue(user))
                .onErrorResume(this::handleError);
    }

    /**
     * Manejo de errores
     */
    private Mono<ServerResponse> handleError(Throwable error) {
        log.error("Error en UserHandler: {}", error.getMessage());

        String code = "INTERNAL_ERROR";
        String message = error.getMessage();
        int status = 500;

        if (error instanceof IllegalArgumentException) {
            code = "VALIDATION_ERROR";
            status = 400;
        } else if (error.getClass().getSimpleName().contains("UsernameAlreadyInUse")) {
            code = "USERNAME_ALREADY_IN_USE";
            status = 409;
        } else if (error.getClass().getSimpleName().contains("InvalidCredentials")) {
            code = "UNAUTHORIZED";
            status = 401;
        }

        return ServerResponse.status(status)
                .bodyValue(java.util.Map.of(
                        "code", code,
                        "message", message
                ));
    }

    /**
     * GET /users/{username}
     * Perfil publico
     */
    public Mono<ServerResponse> getUserByUsername(ServerRequest request) {
        String username = request.pathVariable("username");
        String authHeader = request.headers().firstHeader("Authorization");
        String token = authHeader != null && authHeader.startsWith("Bearer ")
                ? authHeader.substring(7)
                : null;

        return userService.getUserByUsername(username, token)
                .flatMap(user -> ServerResponse.ok().bodyValue(user))
                .onErrorResume(error -> {
                    Throwable cause = Optional.ofNullable(error.getCause()).orElse(error);

                    if (cause instanceof UserNotFoundException) {
                        log.error("Error obteniendo usuario publico", cause);
                        return ServerResponse.status(404)
                                .bodyValue(Map.of(
                                        "code", "USER_NOT_FOUND",
                                        "message", "Usuario no encontrado"
                                ));
                    }

                    if (cause instanceof InvalidCredentialsException) {
                        return ServerResponse.status(401).bodyValue(Map.of(
                                "code", "UNAUTHORIZED",
                                "message", "Token invalido o expirado"
                        ));
                    }

                    log.error("Error inesperado obteniendo usuario publico", cause);
                    return ServerResponse.status(500)
                            .bodyValue(Map.of(
                                    "code", "INTERNAL_ERROR",
                                    "message", "No se pudo obtener el perfil"
                            ));
                });
    }
}
