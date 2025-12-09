package ec.ups.upsglam.post.domain.media.service;

import ec.ups.upsglam.post.domain.media.dto.TempImageResponse;
import ec.ups.upsglam.post.domain.media.dto.UploadImageResponse;
import ec.ups.upsglam.post.infrastructure.pycuda.PyCudaClient;
import ec.ups.upsglam.post.infrastructure.supabase.SupabaseStorageClient;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.io.ByteArrayOutputStream;
import java.time.Instant;
import java.util.UUID;

/**
 * Servicio para procesamiento de imágenes y filtros
 */
@Service
@RequiredArgsConstructor
public class ImageService {

    private static final Logger log = LoggerFactory.getLogger(ImageService.class);
    private final SupabaseStorageClient storageClient;
    private final PyCudaClient pyCudaClient;

    /**
     * Procesa imagen con filtro y la almacena temporalmente
     * FLUJO:
     * 1. Lee bytes de la imagen
     * 2. Llama a PyCUDA service para aplicar filtro
     * 3. Sube imagen filtrada a Supabase en carpeta temp/
     * 4. Devuelve URL temporal + tempImageId
     */
    public Mono<TempImageResponse> processAndStoreTemp(FilePart imagePart, String filter, String userId) {
        log.info("Processing image with filter: {} for user: {}", filter, userId);

        // Generar ID temporal único
        String tempImageId = "temp-" + userId + "-" + UUID.randomUUID().toString();
        String timestamp = String.valueOf(Instant.now().toEpochMilli());
        String fileName = tempImageId + "-" + timestamp + ".jpg";

        // 1. Leer bytes de la imagen original
        return DataBufferUtils.join(imagePart.content())
                .flatMap(dataBuffer -> {
                    byte[] originalBytes = new byte[dataBuffer.readableByteCount()];
                    dataBuffer.read(originalBytes);
                    DataBufferUtils.release(dataBuffer);
                    
                    log.info("Original image bytes read: {} bytes", originalBytes.length);
                    
                    // 2. Aplicar filtro con PyCUDA
                    return pyCudaClient.applyFilter(originalBytes, filter)
                            .flatMap(filteredBytes -> {
                                log.info("Filter applied successfully, uploading to temp storage: {}", fileName);
                                
                                // 3. Subir imagen filtrada a Supabase temp/
                                return storageClient.uploadTempImage(fileName, filteredBytes)
                                        .map(imageUrl -> TempImageResponse.builder()
                                                .tempImageId(tempImageId)
                                                .imageUrl(imageUrl)
                                                .filter(filter)
                                                .build());
                            });
                })
                .doOnSuccess(response -> log.info("Temp image created: {} at {}", response.getTempImageId(), response.getImageUrl()))
                .doOnError(error -> log.error("Error processing image with filter {}: {}", filter, error.getMessage()));
    }

    /**
     * Sube imagen directamente a Supabase Storage
     */
    public Mono<UploadImageResponse> uploadImage(FilePart imagePart, String userId) {
        log.info("Uploading image for user: {} - File: {}", userId, imagePart.filename());

        // Generar nombre único: userId-timestamp-originalName
        String timestamp = String.valueOf(Instant.now().toEpochMilli());
        String originalFilename = imagePart.filename();
        String extension = originalFilename.substring(originalFilename.lastIndexOf("."));
        String fileName = userId + "-" + timestamp + extension;

        // Leer bytes del FilePart
        return DataBufferUtils.join(imagePart.content())
                .flatMap(dataBuffer -> {
                    byte[] bytes = new byte[dataBuffer.readableByteCount()];
                    dataBuffer.read(bytes);
                    DataBufferUtils.release(dataBuffer);
                    
                    log.info("Image bytes read: {} bytes, uploading to Supabase as: {}", bytes.length, fileName);
                    
                    // Subir a Supabase Storage en carpeta posts/
                    return storageClient.uploadPostImage(fileName, bytes);
                })
                .map(imageUrl -> {
                    log.info("Image uploaded successfully: {}", imageUrl);
                    return UploadImageResponse.builder()
                            .imageId(fileName)
                            .imageUrl(imageUrl)
                            .build();
                });
    }

    /**
     * Mueve imagen temporal a carpeta definitiva de posts
     * FLUJO:
     * 1. Obtiene imagen de temp/{tempImageId}
     * 2. Mueve/copia a posts/ con nuevo nombre basado en postId
     * 3. Borra imagen temporal
     * 4. Devuelve URL final
     * 
     * Note: tempImageId es la URL temporal completa de Supabase
     */
    public Mono<String> moveTempToPost(String tempImageUrl, String postId) {
        log.info("Moving temp image from URL {} to post {}", tempImageUrl, postId);

        // Extraer nombre de archivo desde la URL
        // URL ejemplo: https://.../temp/temp-userId-uuid-timestamp.jpg
        String tempPath = storageClient.extractPathFromUrl(tempImageUrl);
        if (tempPath == null || !tempPath.startsWith("temp/")) {
            log.error("Invalid temp image URL: {}", tempImageUrl);
            return Mono.error(new IllegalArgumentException("Invalid temp image URL"));
        }
        
        // Remover "temp/" prefix para obtener solo el nombre del archivo
        final String tempFileName = tempPath.substring(5); // "temp/".length() = 5
        
        String postFileName = postId + ".jpg";
        
        // Supabase tiene método moveTempToPost que hace todo esto
        return storageClient.moveTempToPost(tempFileName, postFileName)
                .doOnSuccess(finalUrl -> log.info("Image moved successfully from temp to posts: {}", finalUrl))
                .doOnError(error -> log.error("Error moving temp image {} to post {}: {}", tempFileName, postId, error.getMessage()));
    }

    /**
     * Borra imagen de Storage
     */
    public Mono<Void> deleteImage(String imageUrl) {
        log.info("Deleting image: {}", imageUrl);

        String path = storageClient.extractPathFromUrl(imageUrl);
        if (path == null || path.isEmpty()) {
            log.warn("Could not extract path from URL: {}", imageUrl);
            return Mono.empty();
        }
        
        return storageClient.deleteFile(path)
                .doOnSuccess(v -> log.info("Image deleted successfully: {}", path))
                .doOnError(error -> log.error("Error deleting image {}: {}", path, error.getMessage()));
    }
}
