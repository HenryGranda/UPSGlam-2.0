package ec.ups.upsglam.auth.domain.exception;

/**
 * Excepción cuando el usuario no es encontrado
 */
public class UserNotFoundException extends RuntimeException {
    public UserNotFoundException(String userId) {
        super("Usuario no encontrado: " + userId);
    }
}
