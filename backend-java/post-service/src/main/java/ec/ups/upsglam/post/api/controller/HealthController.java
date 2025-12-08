package ec.ups.upsglam.post.api.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.util.Map;

/**
 * Controlador simple para verificar que el servidor funciona
 */
@RestController
public class HealthController {

    @GetMapping("/health")
    public Mono<Map<String, String>> health() {
        return Mono.just(Map.of(
                "status", "UP",
                "service", "post-service"
        ));
    }

    @GetMapping("/")
    public Mono<Map<String, String>> root() {
        return Mono.just(Map.of(
                "message", "UPSGlam Post Service API",
                "version", "2.0",
                "endpoints", "/health, /feed, /posts"
        ));
    }
}
