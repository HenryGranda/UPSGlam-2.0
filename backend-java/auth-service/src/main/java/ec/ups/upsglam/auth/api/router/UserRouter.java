package ec.ups.upsglam.auth.api.router;

import ec.ups.upsglam.auth.api.handler.UserHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;

/**
 * Router para endpoints de usuario
 */
@Configuration
public class UserRouter {

        @Bean
        public RouterFunction<ServerResponse> userRoutes(UserHandler handler) {
            return RouterFunctions.route()
                    .PATCH("/users/me", handler::updateProfile)

                    .GET("/users/{username}", handler::getUserByUsername)

                    .build();
        }
}
