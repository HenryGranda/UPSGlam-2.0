package ec.ups.upsglam.post.domain.exception;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.bind.support.WebExchangeBindException;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Manejador global de excepciones
 * Convierte excepciones a respuestas HTTP JSON consistentes
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    @ExceptionHandler(PostNotFoundException.class)
    public ResponseEntity<ErrorResponse> handlePostNotFound(PostNotFoundException ex) {
        log.error("Post no encontrado: {}", ex.getMessage());
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(new ErrorResponse("POST_NOT_FOUND", ex.getMessage()));
    }

    @ExceptionHandler(TempImageNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleTempImageNotFound(TempImageNotFoundException ex) {
        log.error("Imagen temporal no encontrada: {}", ex.getMessage());
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(new ErrorResponse("TEMP_IMAGE_NOT_FOUND", ex.getMessage()));
    }

    @ExceptionHandler(UnauthorizedException.class)
    public ResponseEntity<ErrorResponse> handleUnauthorized(UnauthorizedException ex) {
        log.error("No autorizado: {}", ex.getMessage());
        return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                .body(new ErrorResponse("UNAUTHORIZED", ex.getMessage()));
    }

    @ExceptionHandler(ImageProcessingException.class)
    public ResponseEntity<ErrorResponse> handleImageProcessing(ImageProcessingException ex) {
        log.error("Error procesando imagen: {}", ex.getMessage(), ex);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ErrorResponse("IMAGE_PROCESSING_ERROR", ex.getMessage()));
    }

    @ExceptionHandler(WebExchangeBindException.class)
    public ResponseEntity<Map<String, Object>> handleValidationExceptions(WebExchangeBindException ex) {
        Map<String, Object> errors = new HashMap<>();
        errors.put("code", "VALIDATION_ERROR");
        errors.put("message", "Error de validación");
        
        Map<String, String> fieldErrors = new HashMap<>();
        ex.getBindingResult().getAllErrors().forEach((error) -> {
            String fieldName = ((FieldError) error).getField();
            String errorMessage = error.getDefaultMessage();
            fieldErrors.put(fieldName, errorMessage);
        });
        errors.put("fields", fieldErrors);
        
        log.error("Error de validación: {}", fieldErrors);
        return ResponseEntity.badRequest().body(errors);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGenericException(Exception ex) {
        log.error("Error inesperado: {}", ex.getMessage(), ex);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ErrorResponse("INTERNAL_SERVER_ERROR", 
                    "Ha ocurrido un error inesperado. Por favor, intenta de nuevo."));
    }

    /**
     * Clase interna para respuestas de error
     */
    public static class ErrorResponse {
        public String code;
        public String message;
        public LocalDateTime timestamp;

        public ErrorResponse(String code, String message) {
            this.code = code;
            this.message = message;
            this.timestamp = LocalDateTime.now();
        }
    }
}
