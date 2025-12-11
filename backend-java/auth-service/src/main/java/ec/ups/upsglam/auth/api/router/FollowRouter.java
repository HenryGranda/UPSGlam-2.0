package ec.ups.upsglam.auth.api.router;

import ec.ups.upsglam.auth.api.handler.FollowHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;

/**
 * Router para endpoints de follows
 */
@Configuration
public class FollowRouter {

    @Bean("followRoutesBean")
    public RouterFunction<ServerResponse> followRoutes(FollowHandler handler) {
        return RouterFunctions.route()
                // POST /follows - Seguir a un usuario
                .POST("/follows", handler::followUser)
                
                // DELETE /follows/{userId} - Dejar de seguir
                .DELETE("/follows/{userId}", handler::unfollowUser)
                
                // GET /follows/{userId}/stats - Estadísticas de follows
                .GET("/follows/{userId}/stats", handler::getFollowStats)
                
                // GET /follows/{userId}/followers - Lista de seguidores
                .GET("/follows/{userId}/followers", handler::getFollowers)
                
                // GET /follows/{userId}/following - Lista de seguidos
                .GET("/follows/{userId}/following", handler::getFollowing)
                
                .build();
    }
}
