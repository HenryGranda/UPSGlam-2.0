package ec.ups.upsglam.post.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.config.CorsRegistry;
import org.springframework.web.reactive.config.EnableWebFlux;
import org.springframework.web.reactive.config.WebFluxConfigurer;

/**
 * Configuración de Spring WebFlux
 * - CORS para permitir peticiones desde la app móvil
 * - Configuración de codecs para archivos grandes
 */
@Configuration
@EnableWebFlux
public class WebFluxConfig implements WebFluxConfigurer {

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOrigins("*") // En producción, especificar dominios exactos
                .allowedMethods("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS")
                .allowedHeaders("*")
                .exposedHeaders("Authorization", "X-User-Id", "X-User-Email")
                .maxAge(3600);
    }
}
