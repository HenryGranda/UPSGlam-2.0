package ec.ups.upsglam.post.api.router;

import ec.ups.upsglam.post.api.handler.PostHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Configuration
public class PostRouter {

    @Bean
    public RouterFunction<ServerResponse> postRoutes(PostHandler handler) {
        return route(GET("/feed"), handler::getFeed)
            .andRoute(GET("/posts/{postId}"), handler::getPostById)
            .andRoute(GET("/posts/user/{userId}"), handler::getUserPosts)
            .andRoute(POST("/posts")
                .and(contentType(MediaType.APPLICATION_JSON)), handler::createPost)
            .andRoute(DELETE("/posts/{postId}"), handler::deletePost)
            .andRoute(PATCH("/posts/{postId}/caption"), handler::updateCaption);
    }
}
