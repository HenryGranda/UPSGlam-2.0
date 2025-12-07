package ec.ups.upsglam.post.domain.post.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * DTO para respuesta paginada del feed
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FeedResponse {

    private List<PostResponse> posts;
    private Boolean hasMore;
}
