package ec.ups.upsglam.auth.domain.exception;

/**
 * Excepción lanzada cuando un usuario intenta seguirse a sí mismo
 */
public class SelfFollowException extends RuntimeException {
    public SelfFollowException() {
        super("No puedes seguirte a ti mismo");
    }
}
