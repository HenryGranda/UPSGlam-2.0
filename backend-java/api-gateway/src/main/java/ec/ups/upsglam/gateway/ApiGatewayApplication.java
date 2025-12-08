package ec.ups.upsglam.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * UPSGlam API Gateway
 * 
 * Punto de entrada único para todos los microservicios:
 * - Post Service (http://localhost:8081)
 * - CUDA Service (http://localhost:5000)
 * 
 * El Gateway escucha en http://localhost:8080
 * 
 * Rutas:
 * - /api/posts/** → Post Service
 * - /api/feed → Post Service
 * - /api/images/** → Post Service
 * - /api/filters/** → CUDA Service
 * - /api/health/posts → Health del Post Service
 * - /api/health/cuda → Health del CUDA Service
 * 
 * @author UPSGlam Team
 * @version 2.0
 */
@SpringBootApplication
public class ApiGatewayApplication {

    public static void main(String[] args) {
        SpringApplication.run(ApiGatewayApplication.class, args);
    }
}
