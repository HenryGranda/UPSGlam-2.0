package ec.ups.upsglam.post.domain.exception;

/**
 * Excepción cuando no se encuentra un post
 */
public class PostNotFoundException extends RuntimeException {
    public PostNotFoundException(String postId) {
        super("Post no encontrado: " + postId);
    }
    
    public PostNotFoundException(String message, Throwable cause) {
        super(message, cause);
    }
}
