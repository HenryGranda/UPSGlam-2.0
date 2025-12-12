package ec.ups.upsglam.post.domain.post.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Column;
import org.springframework.data.relational.core.mapping.Table;

import java.time.LocalDateTime;

/**
 * Entidad Post - Tabla posts en Supabase Postgres
 * 
 * Representa una publicación en el feed
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Table("posts")
public class Post {

    @Id
    private String id;

    @Column("user_id")
    private String userId;

    @Column("username")
    private String username;

    @Column("user_photo_url")
    private String userPhotoUrl;

    @Column("image_url")
    private String imageUrl;

    @Column("filter")
    private String filter;

    @Column("audio_file")
    private String audioFile;

    @Column("description")
    private String description;

    @Column("created_at")
    private LocalDateTime createdAt;

    @Column("likes_count")
    private Integer likesCount;

    @Column("comments_count")
    private Integer commentsCount;
}
