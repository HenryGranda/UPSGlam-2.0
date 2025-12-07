package ec.ups.upsglam.post.domain.exception;

/**
 * Excepción cuando el usuario no tiene permisos
 */
public class UnauthorizedException extends RuntimeException {
    public UnauthorizedException(String message) {
        super(message);
    }
}
