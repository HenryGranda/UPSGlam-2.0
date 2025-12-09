package ec.ups.upsglam.post.api.router;

import ec.ups.upsglam.post.api.handler.LikeHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Configuration
public class LikeRouter {

    @Bean
    public RouterFunction<ServerResponse> likeRoutes(LikeHandler handler) {
        return route(POST("/posts/{postId}/likes"), handler::likePost)
            .andRoute(DELETE("/posts/{postId}/likes"), handler::unlikePost)
            .andRoute(GET("/posts/{postId}/likes"), handler::getPostLikes);
    }
}
