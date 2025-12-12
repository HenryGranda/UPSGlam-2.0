// lib/screens/home/home_feed_view.dart
import 'package:flutter/material.dart';

import '../../models/current_user.dart';
import '../../models/post_model.dart';
import 'ui_helpers.dart';

/// Colores inspirados en el logo de la UPS
const kUpsDarkBlue = Color(0xFF003A70);
const kUpsLightBlue = Color(0xFF4F8EC7);
const kUpsYellow   = Color(0xFFF4C430);

class _PlaceholderCenter extends StatelessWidget {
  final String text;
  const _PlaceholderCenter({required this.text});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Text(
        text,
        textAlign: TextAlign.center,
        style: const TextStyle(color: Colors.grey),
      ),
    );
  }
}

/// HOME FEED
class HomeFeedView extends StatelessWidget {
  final List<PostModel> posts;
  final CurrentUser? currentUser;
  final void Function(PostModel) onOpenPost;
  final void Function(String postId) onToggleLike;

  const HomeFeedView({
    super.key,
    required this.posts,
    required this.onOpenPost,
    required this.onToggleLike,
    this.currentUser,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // AppBar custom con colores UPS
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [
                kUpsDarkBlue,
                kUpsLightBlue,
              ],
              begin: Alignment.centerLeft,
              end: Alignment.centerRight,
            ),
            border: Border(
              bottom: BorderSide(color: Color(0xFFE5E7EB)),
            ),
          ),
          child: Row(
            children: [
              Row(
                children: [
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.1),
                      shape: BoxShape.circle,
                    ),
                    padding: const EdgeInsets.all(4),
                    child: Image.asset(
                      'assets/images/logo_upsglam_small.png',
                      height: 28,
                      errorBuilder: (_, __, ___) =>
                          const Icon(Icons.camera_alt_outlined, color: Colors.white),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    'UPSGlam',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: kUpsYellow,
                      letterSpacing: 0.5,
                    ),
                  ),
                ],
              ),
              const Spacer(),
              const Text(
                'Inicio',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
              const Spacer(),
              IconButton(
                onPressed: () {},
                icon: const Icon(
                  Icons.notifications_none,
                  color: kUpsYellow,
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: Container(
            color: const Color(0xFFF5F7FB), // fondo clarito general
            child: posts.isEmpty
                ? const _PlaceholderCenter(text: 'Fin del contenido')
                : ListView.builder(
                    padding: EdgeInsets.zero,
                    itemCount: posts.length + 1,
                    itemBuilder: (context, index) {
                      if (index == posts.length) {
                        return const Padding(
                          padding: EdgeInsets.symmetric(vertical: 24),
                          child: Center(
                            child: Text(
                              'Fin del contenido',
                              style: TextStyle(fontSize: 12, color: Colors.grey),
                            ),
                          ),
                        );
                      }
                      final post = posts[index];
                      return PostCard(
                        post: post,
                        onOpen: () => onOpenPost(post),
                        onToggleLike: () => onToggleLike(post.id),
                      );
                    },
                  ),
          ),
        ),
      ],
    );
  }
}

/// Tarjeta individual de publicación
class PostCard extends StatelessWidget {
  final PostModel post;
  final VoidCallback onOpen;
  final VoidCallback onToggleLike;

  const PostCard({
    super.key,
    required this.post,
    required this.onOpen,
    required this.onToggleLike,
  });

  @override
  Widget build(BuildContext context) {
    final displayUsername = formatUsername(post.username);

    return Card(
      margin: EdgeInsets.zero,
      elevation: 0,
      color: Colors.white,
      shape: const Border(
        bottom: BorderSide(color: Color(0xFFE5E7EB)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // header usuario
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            child: Row(
              children: [
                CircleAvatar(
                  radius: 20,
                  backgroundColor: kUpsLightBlue.withOpacity(0.15),
                  backgroundImage: buildImageProvider(post.avatar),
                  onBackgroundImageError: (_, __) {},
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        displayUsername,
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 13,
                          color: kUpsDarkBlue,
                        ),
                      ),
                      Text(
                        post.timestamp,
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          // imagen
          GestureDetector(
            onTap: onOpen,
            child: AspectRatio(
              aspectRatio: 4 / 5,
              child: Image(
                image: buildImageProvider(post.imageUrl),
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) => Container(
                  color: Colors.grey.shade300,
                  child: const Icon(Icons.image, size: 40),
                ),
              ),
            ),
          ),
          // acciones
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    InkWell(
                      onTap: onToggleLike,
                      child: Row(
                        children: [
                          Icon(
                            post.liked
                                ? Icons.favorite
                                : Icons.favorite_border_outlined,
                            color: post.liked ? kUpsYellow : kUpsDarkBlue,
                            size: 26,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            '${post.likes}',
                            style: const TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w500,
                              color: kUpsDarkBlue,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 16),
                    InkWell(
                      onTap: onOpen,
                      child: Row(
                        children: [
                          const Icon(
                            Icons.mode_comment_outlined,
                            size: 24,
                            color: kUpsDarkBlue,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            '${post.comments.length}',
                            style: const TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w500,
                              color: kUpsDarkBlue,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 16),
                    const Icon(
                      Icons.send_outlined,
                      size: 24,
                      color: kUpsDarkBlue,
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                RichText(
                  text: TextSpan(
                    style: const TextStyle(
                      fontSize: 13,
                      color: Colors.black87,
                    ),
                    children: [
                      TextSpan(
                        text: displayUsername,
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          color: kUpsDarkBlue,
                        ),
                      ),
                      const TextSpan(text: ' '),
                      TextSpan(
                        text: post.caption,
                        style: const TextStyle(
                          color: Colors.black54,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
