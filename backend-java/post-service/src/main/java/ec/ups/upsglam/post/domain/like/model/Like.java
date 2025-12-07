package ec.ups.upsglam.post.domain.like.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Column;
import org.springframework.data.relational.core.mapping.Table;

import java.time.LocalDateTime;

/**
 * Entidad Like - Tabla post_likes en Supabase Postgres
 * PK compuesta: (post_id, user_id)
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Table("post_likes")
public class Like {

    @Id
    @Column("post_id")
    private String postId;

    @Column("user_id")
    private String userId;

    @Column("created_at")
    private LocalDateTime createdAt;
}
