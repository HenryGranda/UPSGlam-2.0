package ec.ups.upsglam.post.api.handler;

import ec.ups.upsglam.post.domain.media.service.ImageService;
import ec.ups.upsglam.post.infrastructure.supabase.SupabaseStorageClient;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.MediaType;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.Map;

@Component
@RequiredArgsConstructor
public class MediaHandler {

    private static final Logger log = LoggerFactory.getLogger(MediaHandler.class);
    private final ImageService imageService;
    private final SupabaseStorageClient storageClient;

    /**
     * POST /images/preview
     * Aplica filtro a imagen y devuelve preview temporal
     * Body: multipart/form-data con "image" y "filter"
     */
    public Mono<ServerResponse> previewFilter(ServerRequest request) {
        String userId = extractUserId(request);

        log.info("Processing image preview for user: {}", userId);

        return request.multipartData()
                .flatMap(multipartData -> {
                    // Extraer la imagen
                    FilePart imagePart = (FilePart) multipartData.getFirst("image");
                    
                    // Extraer el filtro (FormFieldPart, no FilePart)
                    var filterPart = multipartData.getFirst("filter");
                    if (filterPart == null || imagePart == null) {
                        return Mono.error(new IllegalArgumentException("Image and filter are required"));
                    }
                    
                    // Leer el valor del filtro desde el FormFieldPart
                    return DataBufferUtils.join(filterPart.content())
                            .map(dataBuffer -> {
                                byte[] bytes = new byte[dataBuffer.readableByteCount()];
                                dataBuffer.read(bytes);
                                DataBufferUtils.release(dataBuffer);
                                return new String(bytes);
                            })
                            .flatMap(filter -> {
                                log.info("Applying filter: {} to image", filter);
                                return imageService.processAndStoreTemp(imagePart, filter, userId);
                            });
                })
                .flatMap(tempImageResponse ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(tempImageResponse)
                )
                .onErrorResume(this::handleError);
    }

    /**
     * POST /images/upload
     * Sube imagen directamente sin filtro (para avatares, etc)
     * Body: multipart/form-data con "image"
     */
    public Mono<ServerResponse> uploadImage(ServerRequest request) {
        String userId = extractUserId(request);

        log.info("Uploading image for user: {}", userId);

        return request.multipartData()
                .flatMap(multipartData -> {
                    FilePart imagePart = (FilePart) multipartData.getFirst("image");

                    if (imagePart == null) {
                        return Mono.error(new IllegalArgumentException("Image file is required"));
                    }

                    return imageService.uploadImage(imagePart, userId);
                })
                .flatMap(uploadResponse ->
                        ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(uploadResponse)
                )
                .onErrorResume(this::handleError);
    }

    /**
     * Extrae el userId del header X-User-Id
     */
    private String extractUserId(ServerRequest request) {
        return request.headers()
                .firstHeader("X-User-Id");
    }

    /**
     * Manejo centralizado de errores
     */
    private Mono<ServerResponse> handleError(Throwable error) {
        log.error("Error in MediaHandler: ", error);
        
        int statusCode = 500;
        String errorCode = "INTERNAL_SERVER_ERROR";
        
        if (error instanceof IllegalArgumentException) {
            statusCode = 400;
            errorCode = "BAD_REQUEST";
        }
        
        return ServerResponse.status(statusCode)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "error", errorCode,
                        "message", error.getMessage()
                ));
    }
}
