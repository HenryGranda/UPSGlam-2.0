package ec.ups.upsglam.auth.infrastructure.firebase;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

/**
 * Cliente para autenticar usuarios usando la REST API de Firebase Auth
 */
@Component
@Slf4j
public class FirebaseAuthRestClient {

    private final WebClient webClient;
    private final String apiKey;

    public FirebaseAuthRestClient(
            WebClient.Builder webClientBuilder,
            @Value("${firebase.api-key}") String apiKey) {
        this.webClient = webClientBuilder
                .baseUrl("https://identitytoolkit.googleapis.com/v1")
                .build();
        this.apiKey = apiKey;
    }

    /**
     * Autenticar usuario con email y password
     * Retorna un ID Token real de Firebase
     */
    public Mono<FirebaseAuthResponse> signInWithPassword(String email, String password) {
        SignInRequest request = new SignInRequest(email, password, true);
        
        return webClient.post()
                .uri("/accounts:signInWithPassword?key={apiKey}", apiKey)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(FirebaseAuthResponse.class)
                .doOnSuccess(response -> log.info("Usuario autenticado exitosamente: {}", email))
                .doOnError(error -> {
                    log.error("Error autenticando usuario: {}", error.getMessage());
                    if (error instanceof org.springframework.web.reactive.function.client.WebClientResponseException) {
                        org.springframework.web.reactive.function.client.WebClientResponseException webError = 
                            (org.springframework.web.reactive.function.client.WebClientResponseException) error;
                        log.error("Response body: {}", webError.getResponseBodyAsString());
                    }
                });
    }

    @Data
    private static class SignInRequest {
        private final String email;
        private final String password;
        private final boolean returnSecureToken;
    }

    @Data
    public static class FirebaseAuthResponse {
        @JsonProperty("idToken")
        private String idToken;
        
        @JsonProperty("email")
        private String email;
        
        @JsonProperty("refreshToken")
        private String refreshToken;
        
        @JsonProperty("expiresIn")
        private String expiresIn;
        
        @JsonProperty("localId")
        private String localId;
        
        @JsonProperty("registered")
        private boolean registered;
    }
}
