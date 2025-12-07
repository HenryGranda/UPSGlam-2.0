package ec.ups.upsglam.post.domain.media.service;

import ec.ups.upsglam.post.domain.media.dto.TempImageResponse;
import ec.ups.upsglam.post.domain.media.dto.UploadImageResponse;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.UUID;

/**
 * Servicio para procesamiento de imágenes y filtros
 * Sin terceros: solo simula el procesamiento
 */
@Service
@RequiredArgsConstructor
public class ImageService {

    private static final Logger log = LoggerFactory.getLogger(ImageService.class);

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
     * Sube imagen directamente sin filtro
     * SIN TERCEROS: Solo simula el upload
     */
    public Mono<UploadImageResponse> uploadImage(FilePart imagePart, String userId) {
        log.info("Uploading image for user: {}", userId);

        String imageId = UUID.randomUUID().toString();
        String imageUrl = "https://storage.example.com/images/" + imageId + ".jpg";

        // TODO: Cuando se habiliten terceros:
        // 1. Leer bytes de imagePart
        // 2. Subir a Supabase Storage
        // 3. Devolver URL real

        return Mono.just(UploadImageResponse.builder()
                .imageId(imageId)
                .imageUrl(imageUrl)
                .build());
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
