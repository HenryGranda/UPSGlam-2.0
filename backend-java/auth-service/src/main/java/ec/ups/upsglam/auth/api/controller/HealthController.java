package ec.ups.upsglam.auth.api.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.util.Map;

/**
 * Controlador para health check
 */
@RestController
public class HealthController {

    @GetMapping("/health")
    public Mono<Map<String, String>> health() {
        return Mono.just(Map.of(
                "status", "UP",
                "service", "auth-service"
        ));
    }

    @GetMapping("/")
    public Mono<Map<String, Object>> root() {
        return Mono.just(Map.of(
                "service", "UPSGlam Auth Service",
                "version", "2.0",
                "endpoints", Map.of(
                        "register", "POST /api/auth/register",
                        "login", "POST /api/auth/login",
                        "me", "GET /api/auth/me",
                        "updateProfile", "PATCH /api/users/me"
                )
        ));
    }
}
