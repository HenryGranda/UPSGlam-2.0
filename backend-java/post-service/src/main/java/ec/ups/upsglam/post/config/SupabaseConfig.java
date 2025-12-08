package ec.ups.upsglam.post.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

/**
 * Configuración del cliente Supabase
 * - Storage para imágenes (temp, posts, avatars)
 * - Postgres via R2DBC ya está configurado en application.yml
 * 
 * TEMPORALMENTE DESHABILITADO - Sin conexiones externas por ahora
 */
//@Configuration
//@ConfigurationProperties(prefix = "supabase")
@Data
public class SupabaseConfig {

    private String url;
    private String key;
    private String serviceRoleKey;
    private StorageConfig storage;

    @Data
    public static class StorageConfig {
        private String bucket;
        private FoldersConfig folders;
    }

    @Data
    public static class FoldersConfig {
        private String temp;
        private String posts;
        private String avatars;
    }

    /**
     * WebClient para llamar a Supabase Storage API
     */
    //@Bean(name = "supabaseWebClient")
    public WebClient supabaseWebClient() {
        return WebClient.builder()
                .baseUrl(url)
                .defaultHeader("apikey", key)
                .defaultHeader("Authorization", "Bearer " + serviceRoleKey)
                .build();
    }

    /**
     * Obtener URL base del Storage
     */
    public String getStorageUrl() {
        return url + "/storage/v1/object/public/" + storage.bucket;
    }

    /**
     * Obtener URL base de la API de Storage para uploads
     */
    public String getStorageApiUrl() {
        return url + "/storage/v1/object/" + storage.bucket;
    }
}
