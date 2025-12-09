package ec.ups.upsglam.auth.domain.exception;

/**
 * Excepción cuando el username ya está en uso
 */
public class UsernameAlreadyInUseException extends RuntimeException {
    public UsernameAlreadyInUseException(String username) {
        super("El username ya está en uso: " + username);
    }
}
