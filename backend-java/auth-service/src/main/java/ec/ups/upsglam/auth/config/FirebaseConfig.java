package ec.ups.upsglam.auth.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageOptions;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.cloud.FirestoreClient;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;

import java.io.IOException;
import java.io.InputStream;

/**
 * Configuración de Firebase
 */
@Configuration
@Slf4j
public class FirebaseConfig {

    private final ResourceLoader resourceLoader;

    public FirebaseConfig(ResourceLoader resourceLoader) {
        this.resourceLoader = resourceLoader;
    }

    @Value("${firebase.credentials.path}")
    private String credentialsPath;

    @Value("${firebase.project-id}")
    private String projectId;

    @Value("${firebase.database-id:(default)}")
    private String databaseId;

    @Value("${firebase.storage.bucket:}")
    private String storageBucket;

    @Bean
    public FirebaseApp firebaseApp() throws IOException {
        log.info("Inicializando Firebase App...");
        log.info("Cargando credenciales desde: {}", credentialsPath);
        
        Resource resource = resourceLoader.getResource(credentialsPath);
        InputStream serviceAccount = resource.getInputStream();

        FirebaseOptions.Builder optionsBuilder = FirebaseOptions.builder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .setProjectId(projectId);
        
        // Storage bucket es opcional
        if (storageBucket != null && !storageBucket.isEmpty()) {
            optionsBuilder.setStorageBucket(storageBucket);
            log.info("Storage bucket configurado: {}", storageBucket);
        } else {
            log.info("Storage bucket no configurado (opcional)");
        }

        FirebaseOptions options = optionsBuilder.build();

        if (FirebaseApp.getApps().isEmpty()) {
            FirebaseApp app = FirebaseApp.initializeApp(options);
            log.info("Firebase App inicializado correctamente");
            return app;
        }
        
        return FirebaseApp.getInstance();
    }

    @Bean
    public FirebaseAuth firebaseAuth(FirebaseApp firebaseApp) {
        return FirebaseAuth.getInstance(firebaseApp);
    }

    @Bean
    public Firestore firestore(FirebaseApp firebaseApp) {
        log.info("Inicializando Firestore con database: {}", databaseId);
        return FirestoreClient.getFirestore(firebaseApp, databaseId);
    }

    @Bean
    public Storage storage() throws IOException {
        Resource resource = resourceLoader.getResource(credentialsPath);
        InputStream serviceAccount = resource.getInputStream();
        
        return StorageOptions.newBuilder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .setProjectId(projectId)
                .build()
                .getService();
    }
}
