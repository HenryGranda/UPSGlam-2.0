package ec.ups.upsglam.post.config;

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
 * Configuración de Firebase para Post Service
 * - Firestore: para posts, likes y comments
 * - Storage: para imágenes (temp y posts) - OPCIONAL
 * - Auth: para verificar tokens
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

    @Value("${firebase.database-id}")
    private String databaseId;

    @Value("${firebase.storage.bucket}")
    private String storageBucket;

    @Bean
    public FirebaseApp firebaseApp() throws IOException {
        log.info("Inicializando Firebase App en Post Service...");
        log.info("Cargando credenciales desde: {}", credentialsPath);
        
        Resource resource = resourceLoader.getResource(credentialsPath);
        InputStream serviceAccount = resource.getInputStream();

        FirebaseOptions options = FirebaseOptions.builder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .setProjectId(projectId)
                .setStorageBucket(storageBucket)
                .build();

        if (FirebaseApp.getApps().isEmpty()) {
            FirebaseApp app = FirebaseApp.initializeApp(options);
            log.info("Firebase App inicializado correctamente");
            log.info("Project ID: {}", projectId);
            log.info("Storage Bucket: {}", storageBucket);
            return app;
        }
        
        log.info("Firebase App ya estaba inicializado");
        return FirebaseApp.getInstance();
    }

    @Bean
    public FirebaseAuth firebaseAuth(FirebaseApp firebaseApp) {
        log.info("Inicializando Firebase Auth");
        return FirebaseAuth.getInstance(firebaseApp);
    }

    @Bean
    public Firestore firestore(FirebaseApp firebaseApp) {
        log.info("Inicializando Firestore con database: {}", databaseId);
        Firestore firestore = FirestoreClient.getFirestore(firebaseApp, databaseId);
        log.info("Firestore inicializado correctamente");
        return firestore;
    }

    @Bean
    public Storage storage() throws IOException {
        log.info("Inicializando Firebase Storage (OPCIONAL - para futuro)");
        Resource resource = resourceLoader.getResource(credentialsPath);
        InputStream serviceAccount = resource.getInputStream();
        
        Storage storage = StorageOptions.newBuilder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .setProjectId(projectId)
                .build()
                .getService();
        
        log.info("Firebase Storage inicializado correctamente");
        return storage;
    }
}
