package ec.ups.upsglam.auth.api.router;

import ec.ups.upsglam.auth.api.handler.AuthHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;

/**
 * Router para endpoints de autenticación
 */
@Configuration
public class AuthRouter {

    @Bean
    public RouterFunction<ServerResponse> authRoutes(AuthHandler handler) {
        return RouterFunctions.route()
                .POST("/auth/register", handler::register)
                .POST("/auth/login", handler::login)
                .GET("/auth/me", handler::getMe)
                .build();
    }
}
