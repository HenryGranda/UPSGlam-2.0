package ec.ups.upsglam.auth.domain.exception;

/**
 * Excepción cuando el email ya está en uso
 */
public class EmailAlreadyInUseException extends RuntimeException {
    public EmailAlreadyInUseException(String email) {
        super("El email ya está registrado: " + email);
    }
}
