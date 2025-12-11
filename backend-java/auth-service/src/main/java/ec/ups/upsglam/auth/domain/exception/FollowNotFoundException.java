package ec.ups.upsglam.auth.domain.exception;

/**
 * Excepción lanzada cuando no se encuentra una relación de follow
 */
public class FollowNotFoundException extends RuntimeException {
    public FollowNotFoundException(String followerUserId, String followedUserId) {
        super(String.format("No existe relación de follow entre %s y %s", followerUserId, followedUserId));
    }
}
