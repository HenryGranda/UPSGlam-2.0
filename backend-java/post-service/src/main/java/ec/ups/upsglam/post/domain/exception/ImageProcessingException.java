package ec.ups.upsglam.post.domain.exception;

/**
 * Excepción para errores de procesamiento de imágenes (CUDA)
 */
public class ImageProcessingException extends RuntimeException {
    public ImageProcessingException(String message, Throwable cause) {
        super(message, cause);
    }

    public ImageProcessingException(String message) {
        super(message);
    }
}
