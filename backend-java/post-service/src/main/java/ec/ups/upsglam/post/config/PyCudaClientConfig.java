package ec.ups.upsglam.post.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
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
@Slf4j
@Configuration
public class PyCudaClientConfig {

    @Value("${pycuda.service.url:http://localhost:5000}")
    private String urlFromConfig;
    
    @Value("${pycuda.service.timeout:30000}")
    private long timeout;

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
        // Limpiar URL de cualquier carácter invisible
        String cleanUrl = urlFromConfig != null ? urlFromConfig.trim() : "http://localhost:5000";
        
        log.info("=".repeat(60));
        log.info("PyCUDA WebClient Configuration");
        log.info("URL: '{}'", cleanUrl);
        log.info("URL Length: {}", cleanUrl.length());
        log.info("Timeout: {}ms", timeout);
        log.info("=".repeat(60));
        
        HttpClient httpClient = HttpClient.create()
                .responseTimeout(Duration.ofMillis(timeout));

        return WebClient.builder()
                .baseUrl(cleanUrl)
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .defaultHeader("Accept", MediaType.IMAGE_JPEG_VALUE)
                .build();
    }
}

