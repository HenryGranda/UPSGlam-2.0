        package ec.ups.upsglam.auth.api.handler;

        import ec.ups.upsglam.auth.application.FollowService;
        import lombok.RequiredArgsConstructor;
        import lombok.extern.slf4j.Slf4j;
        import org.springframework.stereotype.Component;
        import org.springframework.web.reactive.function.server.ServerRequest;
        import org.springframework.web.reactive.function.server.ServerResponse;
        import reactor.core.publisher.Mono;

        @Component
        @RequiredArgsConstructor
        @Slf4j
        public class FollowHandler {

        private final FollowService followService;

        public Mono<ServerResponse> follow(ServerRequest request) {
                String auth = request.headers().firstHeader("Authorization");
                if (auth == null || !auth.startsWith("Bearer ")) {
                return ServerResponse.status(401).build();
                }

                String token = auth.substring(7);
                String targetUserId = request.pathVariable("userId");

                return followService.followUser(token, targetUserId)
                        .flatMap(res -> ServerResponse.ok().bodyValue(res));
        }

        public Mono<ServerResponse> unfollow(ServerRequest request) {
                String auth = request.headers().firstHeader("Authorization");
                if (auth == null || !auth.startsWith("Bearer ")) {
                return ServerResponse.status(401).build();
                }

                String token = auth.substring(7);
                String targetUserId = request.pathVariable("userId");

                return followService.unfollowUser(token, targetUserId)
                        .flatMap(res -> ServerResponse.ok().bodyValue(res));
        }
        }
