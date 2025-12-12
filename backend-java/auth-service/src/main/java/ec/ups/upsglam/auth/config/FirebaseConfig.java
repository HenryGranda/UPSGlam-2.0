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

import java.io.IOException;
import java.io.InputStream;

/**
 * Configuración de Firebase
 */
@Configuration
@Slf4j
public class FirebaseConfig {

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
        
        // Remover prefijo "file:" si existe para usar FileInputStream directamente
        String cleanPath = credentialsPath.replace("file:", "");
        InputStream serviceAccount = new java.io.FileInputStream(cleanPath);

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
        
        // Si es el database por defecto, usar getFirestore() sin parámetros
        if (databaseId == null || databaseId.isEmpty() || databaseId.equals("(default)")) {
            log.info("Usando database (default) de Firestore");
            return FirestoreClient.getFirestore(firebaseApp);
        }
        
        // Para databases personalizados, usar FirestoreOptions directamente
        log.info("Usando database personalizado de Firestore: {}", databaseId);
        try {
            String cleanPath = credentialsPath.replace("file:", "");
            InputStream serviceAccount = new java.io.FileInputStream(cleanPath);
            
            com.google.cloud.firestore.FirestoreOptions firestoreOptions = 
                com.google.cloud.firestore.FirestoreOptions.newBuilder()
                    .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                    .setProjectId(projectId)
                    .setDatabaseId(databaseId)
                    .build();
            
            return firestoreOptions.getService();
        } catch (IOException e) {
            log.error("Error inicializando Firestore con database personalizado", e);
            throw new RuntimeException("No se pudo inicializar Firestore: " + e.getMessage());
        }
    }

    @Bean
    public Storage storage() throws IOException {
        String cleanPath = credentialsPath.replace("file:", "");
        InputStream serviceAccount = new java.io.FileInputStream(cleanPath);
        
        return StorageOptions.newBuilder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .setProjectId(projectId)
                .build()
                .getService();
    }
}
