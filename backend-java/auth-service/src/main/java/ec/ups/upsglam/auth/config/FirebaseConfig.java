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
import org.springframework.core.io.ClassPathResource;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

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

    // ✅ Lee credenciales desde classpath:... o file:... o ruta normal
    private GoogleCredentials loadCredentials() throws IOException {
        if (credentialsPath == null || credentialsPath.isBlank()) {
            throw new IllegalStateException("firebase.credentials.path está vacío");
        }

        String path = credentialsPath.trim();

        try (InputStream in = openCredentialsStream(path)) {
            return GoogleCredentials.fromStream(in);
        }
    }

    private InputStream openCredentialsStream(String path) throws IOException {
        if (path.startsWith("classpath:")) {
            String resourceName = path.substring("classpath:".length());
            if (resourceName.startsWith("/")) resourceName = resourceName.substring(1);

            log.info("Cargando credenciales desde CLASSPATH: {}", resourceName);
            return new ClassPathResource(resourceName).getInputStream();
        }

        // soporta file:C:\... o file:/...
        if (path.startsWith("file:")) {
            String cleanPath = path.substring("file:".length());
            log.info("Cargando credenciales desde FILE: {}", cleanPath);
            return new FileInputStream(cleanPath);
        }

        // ruta normal
        log.info("Cargando credenciales desde PATH: {}", path);
        return new FileInputStream(path);
    }

    @Bean
    public FirebaseApp firebaseApp() throws IOException {
        log.info("Inicializando Firebase App...");
        log.info("firebase.credentials.path={}", credentialsPath);

        GoogleCredentials credentials = loadCredentials();

        FirebaseOptions.Builder builder = FirebaseOptions.builder()
                .setCredentials(credentials)
                .setProjectId(projectId);

        if (storageBucket != null && !storageBucket.isEmpty()) {
            builder.setStorageBucket(storageBucket);
            log.info("Storage bucket configurado: {}", storageBucket);
        } else {
            log.info("Storage bucket no configurado (opcional)");
        }

        FirebaseOptions options = builder.build();

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

        if (databaseId == null || databaseId.isEmpty() || databaseId.equals("(default)")) {
            log.info("Usando database (default) de Firestore");
            return FirestoreClient.getFirestore(firebaseApp);
        }

        log.info("Usando database personalizado de Firestore: {}", databaseId);
        try {
            GoogleCredentials credentials = loadCredentials();

            com.google.cloud.firestore.FirestoreOptions firestoreOptions =
                    com.google.cloud.firestore.FirestoreOptions.newBuilder()
                            .setCredentials(credentials)
                            .setProjectId(projectId)
                            .setDatabaseId(databaseId)
                            .build();

            return firestoreOptions.getService();
        } catch (IOException e) {
            log.error("Error inicializando Firestore con database personalizado", e);
            throw new RuntimeException("No se pudo inicializar Firestore: " + e.getMessage(), e);
        }
    }

    @Bean
    public Storage storage() throws IOException {
        GoogleCredentials credentials = loadCredentials();

        return StorageOptions.newBuilder()
                .setCredentials(credentials)
                .setProjectId(projectId)
                .build()
                .getService();
    }
}
