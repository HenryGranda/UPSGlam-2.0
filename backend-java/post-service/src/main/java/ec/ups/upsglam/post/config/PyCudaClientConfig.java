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
 * Configuración del cliente PyCUDA Service (FastAPI)
 * 
 * Configura WebClient para llamadas al servicio de procesamiento GPU.
 * El PyCUDA service aplica filtros de convolución acelerados por CUDA.
 * 
 * Configuración en application.yml:
 * <pre>
 * pycuda:
 *   service:
 *     url: http://localhost:5000
 *     timeout: 30000  # 30 segundos para procesamiento GPU
 * </pre>
 * 
 * @see ec.ups.upsglam.post.infrastructure.pycuda.PyCudaClient
 */
@Configuration
@ConfigurationProperties(prefix = "pycuda.service")
@Data
public class PyCudaClientConfig {

    /**
     * Base URL del PyCUDA service.
     * Default: http://localhost:5000
     */
    private String url = "http://localhost:5000";
    
    /**
     * Timeout para operaciones de filtrado (en milisegundos).
     * Default: 30000ms (30 segundos)
     * 
     * GPU processing is fast (usually <1s), but we allow generous timeout for:
     * - First request (CUDA initialization)
     * - Large images (4K+)
     * - Complex filters (LoG with large masks)
     */
    private long timeout = 30000L;

    /**
     * WebClient configurado para PyCUDA Service.
     * 
     * Characteristics:
     * - Response timeout configured for GPU processing
     * - Accepts and sends images in binary format (JPEG/PNG)
     * - Non-blocking reactive client
     * 
     * @return WebClient instance for PyCUDA service calls
     */
    @Bean(name = "pyCudaWebClient")
    public WebClient pyCudaWebClient() {
        HttpClient httpClient = HttpClient.create()
                .responseTimeout(Duration.ofMillis(timeout));

        return WebClient.builder()
                .baseUrl(url)
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .defaultHeader("Accept", MediaType.IMAGE_JPEG_VALUE)
                .build();
    }
}

