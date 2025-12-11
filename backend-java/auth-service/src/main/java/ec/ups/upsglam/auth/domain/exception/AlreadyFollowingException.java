package ec.ups.upsglam.auth.domain.exception;

/**
 * Excepción lanzada cuando ya existe una relación de follow
 */
public class AlreadyFollowingException extends RuntimeException {
    public AlreadyFollowingException(String followedUserId) {
        super(String.format("Ya estás siguiendo al usuario %s", followedUserId));
    }
}
