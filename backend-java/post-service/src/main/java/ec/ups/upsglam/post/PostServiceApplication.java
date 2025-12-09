package ec.ups.upsglam.post;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.r2dbc.repository.config.EnableR2dbcRepositories;

/**
 * UPSGlam Post Service - Microservicio para posts, media, likes y comentarios
 * 
 * Características:
 * - Spring WebFlux (Reactive)
 * - Supabase Storage para imágenes
 * - Supabase Postgres (R2DBC) para datos
 * - Firebase Auth para autenticación
 * - Integración con PyCUDA Service para filtros
 * 
 * @author UPSGlam Team
 * @version 2.0
 */
@SpringBootApplication
@EnableR2dbcRepositories
public class PostServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(PostServiceApplication.class, args);
    }
}
