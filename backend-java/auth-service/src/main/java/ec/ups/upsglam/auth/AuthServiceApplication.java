package ec.ups.upsglam.auth;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * UPSGlam Auth Service - Microservicio de autenticación y gestión de usuarios
 * 
 * Características:
 * - Firebase Authentication para login/registro
 * - Firebase Firestore para datos de perfil de usuario
 * - Firebase Storage para fotos de perfil (avatares)
 * - Spring WebFlux (Reactive)
 * 
 * Endpoints:
 * - POST /auth/register - Crear cuenta
 * - POST /auth/login - Iniciar sesión
 * - GET /auth/me - Obtener perfil del usuario autenticado
 * - PATCH /users/me - Actualizar perfil
 * - POST /users/me/photo - Subir foto de perfil
 * 
 * @author UPSGlam Team
 * @version 2.0
 */
@SpringBootApplication
public class AuthServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(AuthServiceApplication.class, args);
    }
}
