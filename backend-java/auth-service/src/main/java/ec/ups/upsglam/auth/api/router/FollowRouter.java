package ec.ups.upsglam.auth.api.router;

import ec.ups.upsglam.auth.api.handler.FollowHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;

@Configuration
public class FollowRouter {

    @Bean
    public RouterFunction<ServerResponse> followRoutes(FollowHandler handler) {
        return RouterFunctions.route()
                .POST("/users/{userId}/follow", handler::follow)
                .DELETE("/users/{userId}/follow", handler::unfollow)
                .build();
    }
}
