package ec.ups.upsglam.post.api.router;

import ec.ups.upsglam.post.api.handler.CommentHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Configuration
public class CommentRouter {

    @Bean
    public RouterFunction<ServerResponse> commentsRoutes(CommentHandler handler) {
        return route(GET("/posts/{postId}/comments"), handler::getPostComments)
            .andRoute(
                POST("/posts/{postId}/comments")
                    .and(contentType(MediaType.APPLICATION_JSON)),
                handler::createComment
            )
            .andRoute(
                PUT("/posts/{postId}/comments/{commentId}"),
                handler::updateComment
            )
            .andRoute(
                DELETE("/posts/{postId}/comments/{commentId}"),
                handler::deleteComment
            )
            .andRoute(GET("/users/{userId}/comments"), handler::getUserComments);
    }
}
