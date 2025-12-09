package ec.ups.upsglam.gateway.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.util.Map;

/**
 * Controlador raíz del API Gateway
 */
@RestController
public class GatewayController {

    @GetMapping("/")
    public Mono<Map<String, Object>> root() {
        return Mono.just(Map.of(
                "service", "UPSGlam API Gateway",
                "version", "2.0",
                "description", "Punto de entrada único para todos los microservicios",
                "routes", Map.of(
                        "posts", "/api/posts/**",
                        "feed", "/api/feed",
                        "images", "/api/images/**",
                        "filters", "/api/filters/**",
                        "health", Map.of(
                                "gateway", "/actuator/health",
                                "posts", "/api/health/posts",
                                "cuda", "/api/health/cuda"
                        )
                )
        ));
    }

    @GetMapping("/health")
    public Mono<Map<String, String>> health() {
        return Mono.just(Map.of(
                "status", "UP",
                "service", "api-gateway"
        ));
    }
}
