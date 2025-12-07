package ec.ups.upsglam.post.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

import java.time.Duration;

/**
 * Configuración del cliente PyCUDA Service
 * Para aplicar filtros de convolución en GPU
 * 
 * TEMPORALMENTE DESHABILITADO - Sin conexiones externas por ahora
 */
//@Configuration
//@ConfigurationProperties(prefix = "pycuda.service")
@Data
public class PyCudaClientConfig {

    private String url;
    private long timeout;

    /**
     * WebClient configurado para llamar al PyCUDA Service
     * - Timeout configurado para procesamiento de imágenes
     * - Acepta y envía imágenes en formato binario
     */
    //@Bean(name = "pycudaWebClient")
    public WebClient pycudaWebClient() {
        HttpClient httpClient = HttpClient.create()
                .responseTimeout(Duration.ofMillis(timeout));

        return WebClient.builder()
                .baseUrl(url)
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .defaultHeader("Accept", MediaType.IMAGE_JPEG_VALUE)
                .build();
    }
}
