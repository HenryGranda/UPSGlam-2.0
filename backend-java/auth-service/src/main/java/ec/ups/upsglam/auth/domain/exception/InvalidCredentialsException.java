package ec.ups.upsglam.auth.domain.exception;

/**
 * Excepción cuando las credenciales son inválidas
 */
public class InvalidCredentialsException extends RuntimeException {
    public InvalidCredentialsException() {
        super("Usuario o contraseña incorrectos");
    }
}
