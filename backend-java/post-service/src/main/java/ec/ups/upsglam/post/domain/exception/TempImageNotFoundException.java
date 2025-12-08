package ec.ups.upsglam.post.domain.exception;

/**
 * Excepción cuando no se encuentra una imagen temporal
 */
public class TempImageNotFoundException extends RuntimeException {
    public TempImageNotFoundException(String tempImageId) {
        super("Imagen temporal no encontrada o expirada: " + tempImageId);
    }
}
