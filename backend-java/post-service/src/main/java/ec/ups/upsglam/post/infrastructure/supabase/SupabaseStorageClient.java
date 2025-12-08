package ec.ups.upsglam.post.infrastructure.supabase;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

/**
 * Cliente para interactuar con Supabase Storage
 * Maneja la subida y eliminación de imágenes en buckets de Supabase
 */
@Component
@Slf4j
public class SupabaseStorageClient {

    private final WebClient webClient;
    private final String supabaseUrl;
    private final String bucket;
    private final String postsFolder;
    private final String tempFolder;
    private final String avatarsFolder;

    public SupabaseStorageClient(
            @Value("${supabase.url}") String supabaseUrl,
            @Value("${supabase.service-role-key}") String serviceRoleKey,
            @Value("${supabase.storage.bucket}") String bucket,
            @Value("${supabase.storage.folders.posts}") String postsFolder,
            @Value("${supabase.storage.folders.temp}") String tempFolder,
            @Value("${supabase.storage.folders.avatars}") String avatarsFolder
    ) {
        this.supabaseUrl = supabaseUrl;
        this.bucket = bucket;
        this.postsFolder = postsFolder;
        this.tempFolder = tempFolder;
        this.avatarsFolder = avatarsFolder;

        this.webClient = WebClient.builder()
                .baseUrl(supabaseUrl + "/storage/v1")
                .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + serviceRoleKey)
                .defaultHeader("apikey", serviceRoleKey)
                .build();

        log.info("Supabase Storage Client inicializado");
        log.info("Bucket: {}", bucket);
        log.info("Base URL: {}/storage/v1", supabaseUrl);
    }

    /**
     * Sube una imagen al folder de posts
     * @param fileName nombre del archivo (ej: postId.jpg)
     * @param fileBytes bytes de la imagen
     * @return URL pública de la imagen
     */
    public Mono<String> uploadPostImage(String fileName, byte[] fileBytes) {
        String path = postsFolder + "/" + fileName;
        return uploadFile(path, fileBytes)
                .map(result -> getPublicUrl(path));
    }

    /**
     * Sube una imagen al folder temporal
     * @param fileName nombre del archivo temporal (ej: temp-uuid.jpg)
     * @param fileBytes bytes de la imagen
     * @return URL pública de la imagen temporal
     */
    public Mono<String> uploadTempImage(String fileName, byte[] fileBytes) {
        String path = tempFolder + "/" + fileName;
        return uploadFile(path, fileBytes)
                .map(result -> getPublicUrl(path));
    }

    /**
     * Sube una imagen de avatar
     * @param fileName nombre del archivo (ej: userId.jpg)
     * @param fileBytes bytes de la imagen
     * @return URL pública del avatar
     */
    public Mono<String> uploadAvatarImage(String fileName, byte[] fileBytes) {
        String path = avatarsFolder + "/" + fileName;
        return uploadFile(path, fileBytes)
                .map(result -> getPublicUrl(path));
    }

    /**
     * Elimina una imagen del storage
     * @param path ruta completa del archivo (ej: posts/postId.jpg)
     */
    public Mono<Void> deleteFile(String path) {
        log.debug("Eliminando archivo: {}", path);
        
        // Extraer bucket y path: path puede venir como "posts/file.jpg" o solo "file.jpg"
        String filePath = path.contains("/") ? path : "posts/" + path;
        
        return webClient.delete()
                .uri(uriBuilder -> uriBuilder
                        .path("/object/{bucket}/{+path}")
                        .build(bucket, filePath))
                .retrieve()
                .bodyToMono(String.class)
                .doOnSuccess(response -> log.debug("Archivo eliminado: {}", path))
                .doOnError(error -> log.error("Error eliminando archivo: {}", path, error))
                .then();
    }

    /**
     * Mueve un archivo de temp a posts
     * @param tempFileName nombre del archivo temporal
     * @param postFileName nombre del archivo final
     */
    public Mono<String> moveTempToPost(String tempFileName, String postFileName) {
        String tempPath = tempFolder + "/" + tempFileName;
        String postPath = postsFolder + "/" + postFileName;
        
        log.debug("Moviendo archivo de {} a {}", tempPath, postPath);
        
        // Supabase no tiene "move", así que hacemos: download -> upload -> delete
        return getFileBytes(tempPath)
                .flatMap(bytes -> uploadFile(postPath, bytes))
                .flatMap(result -> deleteFile(tempPath).thenReturn(result))
                .map(result -> getPublicUrl(postPath))
                .doOnSuccess(url -> log.debug("Archivo movido exitosamente a: {}", url))
                .doOnError(error -> log.error("Error moviendo archivo", error));
    }

    /**
     * Sube un archivo al storage
     */
    private Mono<String> uploadFile(String path, byte[] fileBytes) {
        log.debug("Subiendo archivo a: {}", path);
        log.debug("Upload URL: /object/{}/{}", bucket, path);
        
        return webClient.post()
                .uri("/object/{bucket}/{path}", bucket, path)
                .contentType(MediaType.IMAGE_JPEG) // Supabase requiere el tipo correcto
                .header("x-upsert", "true") // Permite sobrescribir si existe
                .bodyValue(fileBytes)
                .retrieve()
                .bodyToMono(String.class)
                .doOnSuccess(response -> log.debug("Archivo subido exitosamente: {} - Response: {}", path, response))
                .doOnError(error -> log.error("Error subiendo archivo: {} - Error: {}", path, error.getMessage()));
    }

    /**
     * Obtiene los bytes de un archivo
     */
    private Mono<byte[]> getFileBytes(String path) {
        return webClient.get()
                .uri("/object/{bucket}/{path}", bucket, path)
                .retrieve()
                .bodyToMono(byte[].class);
    }

    /**
     * Construye la URL pública de un archivo
     */
    private String getPublicUrl(String path) {
        // URL pública de Supabase Storage
        // Formato: https://{project}.supabase.co/storage/v1/object/public/{bucket}/{path}
        return supabaseUrl + "/storage/v1/object/public/" + bucket + "/" + path;
    }

    /**
     * Extrae el path del archivo desde una URL pública
     */
    public String extractPathFromUrl(String url) {
        if (url == null || !url.contains("/public/" + bucket + "/")) {
            return null;
        }
        return url.substring(url.indexOf("/public/" + bucket + "/") + ("/public/" + bucket + "/").length());
    }
}
