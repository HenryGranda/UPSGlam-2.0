package ec.ups.upsglam.post.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.auth.FirebaseAuth;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;

import java.io.IOException;

/**
 * Configuración de Firebase Admin SDK
 * Solo se usa para verificar tokens de autenticación (Firebase Auth)
 * Los datos y storage ahora están en Supabase
 * 
 * TEMPORALMENTE DESHABILITADO - Sin conexiones externas por ahora
 */
//@Configuration
public class FirebaseConfig {

    private static final Logger log = LoggerFactory.getLogger(FirebaseConfig.class);

    @Value("${firebase.credentials.path}")
    private Resource credentialsPath;

    @Value("${firebase.project-id}")
    private String projectId;

    //@Bean
    public FirebaseApp firebaseApp() throws IOException {
        if (FirebaseApp.getApps().isEmpty()) {
            FirebaseOptions options = FirebaseOptions.builder()
                    .setCredentials(GoogleCredentials.fromStream(credentialsPath.getInputStream()))
                    .setProjectId(projectId)
                    .build();

            FirebaseApp app = FirebaseApp.initializeApp(options);
            log.info("Firebase Admin SDK inicializado correctamente para proyecto: {}", projectId);
            return app;
        }
        return FirebaseApp.getInstance();
    }

    //@Bean
    public FirebaseAuth firebaseAuth(FirebaseApp firebaseApp) {
        return FirebaseAuth.getInstance(firebaseApp);
    }
}
