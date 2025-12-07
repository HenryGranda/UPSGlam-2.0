package ec.ups.upsglam.post.domain.comment.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Column;
import org.springframework.data.relational.core.mapping.Table;

import java.time.LocalDateTime;

/**
 * Entidad Comment - Tabla comments en Supabase Postgres
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Table("comments")
public class Comment {

    @Id
    private String id;

    @Column("post_id")
    private String postId;

    @Column("user_id")
    private String userId;

    @Column("username")
    private String username;

    @Column("user_photo_url")
    private String userPhotoUrl;

    @Column("text")
    private String text;

    @Column("created_at")
    private LocalDateTime createdAt;
}
