package ec.ups.upsglam.auth.api.router;

import ec.ups.upsglam.auth.api.handler.NotificationHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;

@Configuration
public class NotificationRouter {

    @Bean
    public RouterFunction<ServerResponse> notificationRoutes(NotificationHandler handler) {
        // spring.webflux.base-path=/api, so route patterns here must omit the /api prefix
        return RouterFunctions.route(GET("/notifications"), handler::list)
                .andRoute(GET("/notifications/unread-count"), handler::unreadCount)
                .andRoute(PATCH("/notifications/{id}/read"), handler::markRead)
                .andRoute(POST("/users/me/fcm-token"), handler::saveFcmToken);
    }
}
