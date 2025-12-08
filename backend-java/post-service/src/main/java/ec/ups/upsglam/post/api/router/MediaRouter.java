package ec.ups.upsglam.post.api.router;

import ec.ups.upsglam.post.api.handler.MediaHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.*;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Configuration
public class MediaRouter {

    @Bean
    public RouterFunction<ServerResponse> imageRoutes(MediaHandler handler) {
        return route(POST("/images/preview")
                .and(contentType(MediaType.MULTIPART_FORM_DATA)), handler::previewFilter)
            .andRoute(POST("/images/upload")
                .and(contentType(MediaType.MULTIPART_FORM_DATA)), handler::uploadImage);
    }
}
