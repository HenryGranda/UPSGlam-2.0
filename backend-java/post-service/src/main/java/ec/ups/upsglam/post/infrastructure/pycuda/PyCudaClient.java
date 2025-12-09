package ec.ups.upsglam.post.infrastructure.pycuda;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;

/**
 * Client for PyCUDA Service (FastAPI)
 * 
 * Calls PyCUDA service to apply CUDA-accelerated convolution filters to images.
 * This service handles GPU processing for 6 filters:
 * - 4 class filters: prewitt, laplacian, gaussian, box_blur
 * - 2 creative UPS filters: ups_logo, ups_color
 * 
 * Architecture: post-service → PyCUDA FastAPI → GPU processing → filtered image bytes
 * 
 * @see backend-java/cuda-lab-back/app.py
 * @see backend-java/README-PYCUDA-GUIDE.MD
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class PyCudaClient {
    
    private final WebClient pyCudaWebClient;
    
    /**
     * Apply filter to image using PyCUDA service.
     * 
     * Flow:
     * 1. Send raw image bytes to PyCUDA service
     * 2. PyCUDA processes on GPU with specified filter
     * 3. Return filtered image bytes (JPEG)
     * 
     * @param imageBytes Raw image bytes (JPEG/PNG)
     * @param filterName Filter to apply (prewitt, laplacian, gaussian, box_blur, ups_logo, ups_color)
     * @return Mono<byte[]> Filtered image bytes (JPEG)
     * 
     * Example:
     * <pre>
     * pyCudaClient.applyFilter(imageBytes, "gaussian")
     *     .flatMap(filteredBytes -> storageClient.uploadPostImage(fileName, filteredBytes))
     *     .subscribe(imageUrl -> log.info("Filtered image uploaded: {}", imageUrl));
     * </pre>
     */
    public Mono<byte[]> applyFilter(byte[] imageBytes, String filterName) {
        log.info("Applying filter '{}' via PyCUDA service (image size: {} bytes)", filterName, imageBytes.length);
        log.info("WebClient base URL: {}", pyCudaWebClient.toString());
        
        try {
            log.info("Starting URI construction for filter: {}", filterName);
            return pyCudaWebClient.post()
                .uri(uriBuilder -> {
                    log.info("Inside URI builder - filterName: {}", filterName);
                    log.info("UriBuilder class: {}", uriBuilder.getClass().getName());
                    log.info("Building path: /filters/{}", filterName);
                    
                    try {
                        var uri = uriBuilder.path("/filters/{filterName}").build(filterName);
                        log.info("Built URI successfully: {}", uri);
                        return uri;
                    } catch (Exception e) {
                        log.error("ERROR building URI inside builder: {}", e.getMessage(), e);
                        throw e;
                    }
                })
                .contentType(MediaType.IMAGE_JPEG)
                .accept(MediaType.IMAGE_JPEG)
                .bodyValue(imageBytes)
                .retrieve()
                .onStatus(
                    status -> status.value() == 400,
                    response -> response.bodyToMono(String.class)
                        .flatMap(body -> {
                            log.error("PyCUDA filter validation error: {}", body);
                            return Mono.error(new PyCudaFilterException("Invalid filter or image: " + body));
                        })
                )
                .onStatus(
                    status -> status.value() == 503,
                    response -> response.bodyToMono(String.class)
                        .flatMap(body -> {
                            log.error("PyCUDA GPU error: {}", body);
                            return Mono.error(new PyCudaGpuException("GPU processing error: " + body));
                        })
                )
                .onStatus(
                    status -> status.is5xxServerError(),
                    response -> response.bodyToMono(String.class)
                    .flatMap(body -> {
                        log.error("PyCUDA service error: {}", body);
                        return Mono.error(new PyCudaServiceException("PyCUDA service error: " + body));
                    })
            )
            .bodyToMono(byte[].class)
            .doOnNext(filteredBytes -> 
                log.info("Filter '{}' applied successfully (output size: {} bytes)", filterName, filteredBytes.length)
            )
            .doOnError(error -> 
                log.error("Error applying filter '{}': {}", filterName, error.getMessage(), error)
            )
            .onErrorMap(
                WebClientResponseException.class,
                ex -> new PyCudaServiceException(
                    String.format("PyCUDA service communication error [%d]: %s", 
                        ex.getRawStatusCode(), ex.getResponseBodyAsString())
                )
            );
        } catch (Exception e) {
            log.error("Exception building request to PyCUDA: {}", e.getMessage(), e);
            return Mono.error(new PyCudaServiceException("Failed to build request: " + e.getMessage(), e));
        }
    }
    
    /**
     * List available filters from PyCUDA service.
     * 
     * @return Mono<String> JSON response with filter catalog
     * 
     * Example response:
     * <pre>
     * {
     *   "filters": [
     *     {"name": "prewitt", "description": "Edge detection", "type": "convolution"},
     *     {"name": "gaussian", "description": "Smoothing", "type": "convolution"},
     *     {"name": "ups_logo", "description": "UPS logo overlay", "type": "creative"}
     *   ]
     * }
     * </pre>
     */
    public Mono<String> listFilters() {
        log.debug("Listing available filters from PyCUDA service");
        
        return pyCudaWebClient.get()
            .uri("/filters")
            .accept(MediaType.APPLICATION_JSON)
            .retrieve()
            .bodyToMono(String.class)
            .doOnSuccess(filters -> log.debug("Available filters: {}", filters))
            .doOnError(error -> log.error("Error listing filters: {}", error.getMessage()));
    }
    
    /**
     * Check PyCUDA service health.
     * 
     * @return Mono<Boolean> True if service is healthy
     */
    public Mono<Boolean> checkHealth() {
        return pyCudaWebClient.get()
            .uri("/health")
            .retrieve()
            .bodyToMono(String.class)
            .map(response -> response.contains("ok"))
            .onErrorReturn(false)
            .doOnNext(healthy -> log.debug("PyCUDA service health: {}", healthy ? "OK" : "DOWN"));
    }
    
    // Custom exceptions
    
    public static class PyCudaFilterException extends RuntimeException {
        public PyCudaFilterException(String message) {
            super(message);
        }
    }
    
    public static class PyCudaGpuException extends RuntimeException {
        public PyCudaGpuException(String message) {
            super(message);
        }
    }
    
    public static class PyCudaServiceException extends RuntimeException {
        public PyCudaServiceException(String message) {
            super(message);
        }
        
        public PyCudaServiceException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
