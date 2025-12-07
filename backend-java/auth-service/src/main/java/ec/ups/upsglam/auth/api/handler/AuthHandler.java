package ec.ups.upsglam.auth.api.handler;

import ec.ups.upsglam.auth.application.AuthService;
import ec.ups.upsglam.auth.domain.dto.LoginRequest;
import ec.ups.upsglam.auth.domain.dto.RegisterRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

/**
 * Handler para endpoints de autenticación
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class AuthHandler {

    private final AuthService authService;

    /**
     * POST /auth/register
     */
    public Mono<ServerResponse> register(ServerRequest request) {
        return request.bodyToMono(RegisterRequest.class)
                .flatMap(authService::register)
                .flatMap(response -> ServerResponse.ok().bodyValue(response))
                .onErrorResume(this::handleError);
    }

    /**
     * POST /auth/login
     */
    public Mono<ServerResponse> login(ServerRequest request) {
        return request.bodyToMono(LoginRequest.class)
                .flatMap(authService::login)
                .flatMap(response -> ServerResponse.ok().bodyValue(response))
                .onErrorResume(this::handleError);
    }

    /**
     * GET /auth/me
     */
    public Mono<ServerResponse> getMe(ServerRequest request) {
        String authHeader = request.headers().firstHeader("Authorization");
        
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return ServerResponse.status(401)
                    .bodyValue(java.util.Map.of(
                            "code", "UNAUTHORIZED",
                            "message", "Token requerido"
                    ));
        }
        
        String token = authHeader.substring(7);
        
        return authService.getMe(token)
                .flatMap(user -> ServerResponse.ok().bodyValue(user))
                .onErrorResume(this::handleError);
    }

    /**
     * Manejo de errores
     */
    private Mono<ServerResponse> handleError(Throwable error) {
        log.error("Error en AuthHandler: {}", error.getMessage());
        
        String code = "INTERNAL_ERROR";
        String message = error.getMessage();
        int status = 500;
        
        if (error instanceof IllegalArgumentException) {
            code = "VALIDATION_ERROR";
            status = 400;
        } else if (error.getClass().getSimpleName().contains("EmailAlreadyInUse")) {
            code = "EMAIL_ALREADY_IN_USE";
            status = 409;
        } else if (error.getClass().getSimpleName().contains("UsernameAlreadyInUse")) {
            code = "USERNAME_ALREADY_IN_USE";
            status = 409;
        } else if (error.getClass().getSimpleName().contains("InvalidCredentials")) {
            code = "INVALID_CREDENTIALS";
            status = 401;
        } else if (error.getClass().getSimpleName().contains("UserNotFound")) {
            code = "USER_NOT_FOUND";
            status = 404;
        }
        
        return ServerResponse.status(status)
                .bodyValue(java.util.Map.of(
                        "code", code,
                        "message", message
                ));
    }
}
