package ec.ups.upsglam.post.domain.media.service;

import ec.ups.upsglam.post.domain.media.dto.TempImageResponse;
import ec.ups.upsglam.post.domain.media.dto.UploadImageResponse;
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

    /**
     * Procesa imagen con filtro y la almacena temporalmente
     * SIN TERCEROS: Solo simula el proceso
     */
    public Mono<TempImageResponse> processAndStoreTemp(FilePart imagePart, String filter, String userId) {
        log.info("Processing image with filter: {} for user: {}", filter, userId);

        // Generar ID temporal
        String tempImageId = "temp-" + UUID.randomUUID().toString();
        String tempImageUrl = "https://storage.example.com/temp/" + tempImageId + ".jpg";

        // TODO: Cuando se habiliten terceros:
        // 1. Leer bytes de imagePart
        // 2. Llamar a PyCUDA service con filter
        // 3. Subir imagen filtrada a Supabase Storage en carpeta temp/
        // 4. Devolver URL real

        return Mono.just(TempImageResponse.builder()
                .tempImageId(tempImageId)
                .imageUrl(tempImageUrl)
                .filter(filter)
                .build());
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
     * SIN TERCEROS: Solo simula el movimiento
     */
    public Mono<String> moveTempToPost(String tempImageId, String postId) {
        log.info("Moving temp image {} to post {}", tempImageId, postId);

        String finalImageUrl = "https://storage.example.com/posts/" + postId + ".jpg";

        // TODO: Cuando se habiliten terceros:
        // 1. Obtener imagen de temp/{tempImageId}
        // 2. Mover/copiar a posts/{postId}
        // 3. Borrar imagen temporal
        // 4. Devolver URL final

        return Mono.just(finalImageUrl);
    }

    /**
     * Borra imagen de Storage
     * SIN TERCEROS: Solo simula el borrado
     */
    public Mono<Void> deleteImage(String imageUrl) {
        log.info("Deleting image: {}", imageUrl);

        // TODO: Cuando se habiliten terceros:
        // 1. Extraer path de la URL
        // 2. Borrar de Supabase Storage

        return Mono.empty();
    }
}
