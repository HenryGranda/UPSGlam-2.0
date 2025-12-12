// lib/screens/home/profile_view.dart
import 'package:flutter/material.dart';

import '../../models/current_user.dart';
import '../../models/post_model.dart';
import '../../models/avatars.dart';
import '../../screens/home/edit_profile_screen.dart';

import 'post_detail_page.dart';
import 'ui_helpers.dart';

/// Colores inspirados en el logo de la UPS
const kUpsDarkBlue = Color.fromARGB(255, 1, 41, 79);
const kUpsLightBlue = Color(0xFF4F8EC7);
const kUpsYellow = Color(0xFFF4C430);

class ProfileView extends StatelessWidget {
  final List<PostModel> posts;
  final CurrentUser? currentUser;
  final Future<void> Function()? onLogout;
  final Future<void> Function()? onProfileUpdated;
  final Future<void> Function()? onRefresh;
  final void Function(String postId)? onDeletePost;
  final void Function(String postId, String newCaption)? onCaptionUpdated;
  final String? viewerUsername;

  // Para distinguir entre mi perfil y el de otros
  final bool isMe;
  final bool isFollowing;
  final Future<void> Function()? onToggleFollow;

  const ProfileView({
    super.key,
    required this.posts,
    this.currentUser,
    this.onLogout,
    this.onProfileUpdated,
    this.onRefresh,
    this.onDeletePost,
    this.onCaptionUpdated,
    this.viewerUsername,
    this.isMe = true,
    this.isFollowing = false,
    this.onToggleFollow,
  });

  @override
  Widget build(BuildContext context) {
    final user = currentUser;

    // USERNAME dinámico
    final username = formatUsername(user?.username);

    // BIO dinámica
    final bio = (user?.bio != null && user!.bio!.trim().isNotEmpty)
        ? user.bio!
        : 'Añade una biografía para que te conozcan mejor ✏️';

    // AVATAR
    final ImageProvider<Object> avatar;
    final photo = user?.photoUrl;

    if (photo != null && photo.isNotEmpty) {
      if (photo.startsWith('http')) {
        avatar = NetworkImage(photo);
      } else {
        final assetPath = avatarAssetFromName(photo);
        avatar = AssetImage(assetPath);
      }
    } else {
      avatar = AssetImage(avatarAssetFromName(kAvatarNames.first));
    }

    // POSTS SOLO DEL USUARIO LOGUEADO
    final userPosts = posts.where((p) {
      if (user == null || user.username.isEmpty) return false;
      final raw = user.username;
      final u1 = raw;
      final u2 = raw.startsWith('@') ? raw : '@$raw';
      return p.username == u1 || p.username == u2;
    }).toList();

    return Container(
      color: const Color(0xFFF5F7FB),
      child: Column(
        children: [
          // HEADER con gradient UPS
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [kUpsDarkBlue, kUpsLightBlue],
                begin: Alignment.centerLeft,
                end: Alignment.centerRight,
              ),
              border: Border(
                bottom: BorderSide(color: Color(0xFFE5E7EB)),
              ),
            ),
            child: Row(
              children: [
                if (!isMe)
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.arrow_back, color: Colors.white),
                  )
                else
                  const SizedBox(width: 48),

                const Spacer(),
                const Text(
                  'Perfil',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: Colors.white,
                  ),
                ),
                const Spacer(),

                if (isMe)
                  IconButton(
                    onPressed: () => _openSettingsSheet(context),
                    icon: const Icon(
                      Icons.settings_outlined,
                      color: kUpsYellow,
                    ),
                  )
                else
                  const SizedBox(width: 48),
              ],
            ),
          ),

          // CONTENIDO
          Expanded(
            child: RefreshIndicator(
              onRefresh: onRefresh ?? () async {},
              child: SingleChildScrollView(
                physics: const AlwaysScrollableScrollPhysics(),
                child: Column(
                  children: [
                    const SizedBox(height: 20),

                    // FOTO con aro de color
                    Container(
                      padding: const EdgeInsets.all(3),
                      decoration: const BoxDecoration(
                        shape: BoxShape.circle,
                        gradient: LinearGradient(
                          colors: [kUpsYellow, kUpsLightBlue],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                      ),
                      child: CircleAvatar(
                        radius: 40,
                        backgroundImage: avatar,
                        backgroundColor: Colors.grey.shade200,
                      ),
                    ),

                    const SizedBox(height: 12),

                    // USERNAME
                    Text(
                      username,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: kUpsDarkBlue,
                      ),
                    ),

                    const SizedBox(height: 4),

                    // BIO
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 32),
                      child: Text(
                        bio,
                        textAlign: TextAlign.center,
                        style: const TextStyle(fontSize: 13, color: Colors.grey),
                      ),
                    ),

                    const SizedBox(height: 12),

                    // BOTÓN EDITAR PERFIL / SEGUIR
                    if (isMe)
                      FilledButton.tonal(
                        onPressed: () => _openSettingsSheet(context),
                        child: const Text('Editar perfil'),
                      )
                    else
                      FilledButton(
                        onPressed: onToggleFollow == null
                            ? null
                            : () => onToggleFollow!(),
                        style: FilledButton.styleFrom(
                          backgroundColor:
                              isFollowing ? Colors.white : kUpsYellow,
                          foregroundColor:
                              isFollowing ? kUpsDarkBlue : Colors.black,
                          side: isFollowing
                              ? const BorderSide(color: kUpsDarkBlue)
                              : BorderSide.none,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 24,
                            vertical: 8,
                          ),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(999),
                          ),
                        ),
                        child: Text(isFollowing ? 'Siguiendo' : 'Seguir'),
                      ),

                    const SizedBox(height: 16),

                    // ESTADISTICAS
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        _ProfileStat(
                          label: 'Publicaciones',
                          value: '${userPosts.length}',
                        ),
                        _ProfileStat(
                          label: 'Seguidores',
                          value: '${user?.followersCount ?? 0}',
                        ),
                        _ProfileStat(
                          label: 'Siguiendo',
                          value: '${user?.followingCount ?? 0}',
                        ),
                      ],
                    ),

                    const SizedBox(height: 16),

                    // TABS PUBLICACIONES / GUARDADAS con colorcito
                    Container(
                      decoration: const BoxDecoration(
                        color: Colors.white,
                        border: Border(
                          top: BorderSide(color: Color(0xFFE5E7EB)),
                          bottom: BorderSide(color: Color(0xFFE5E7EB)),
                        ),
                      ),
                      child: Row(
                        children: [
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              decoration: const BoxDecoration(
                                border: Border(
                                  bottom: BorderSide(
                                    color: kUpsYellow,
                                    width: 2,
                                  ),
                                ),
                              ),
                              child: const Center(
                                child: Text(
                                  'Publicaciones',
                                  style: TextStyle(
                                    fontWeight: FontWeight.w600,
                                    color: kUpsDarkBlue,
                                  ),
                                ),
                              ),
                            ),
                          ),
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              child: const Center(
                                child: Text(
                                  'Guardadas',
                                  style: TextStyle(
                                    fontWeight: FontWeight.w600,
                                    color: Colors.grey,
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),

                    // GRID DE POSTS DEL USUARIO
                    GridView.builder(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      padding: const EdgeInsets.all(1),
                      gridDelegate:
                          const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 3,
                        mainAxisSpacing: 1,
                        crossAxisSpacing: 1,
                      ),
                      itemCount: userPosts.length,
                      itemBuilder: (_, index) {
                        final post = userPosts[index];
                        return GestureDetector(
                          onTap: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (_) => PostDetailPage(
                                  post: post,
                                  currentUser: currentUser,
                                  onToggleLike: (_) {},
                                  onAddComment: (_, __) {},
                                  onDeletePost: onDeletePost,
                                  onCaptionUpdated: onCaptionUpdated,
                                  viewerUsername: viewerUsername,
                                ),
                              ),
                            );
                          },
                          child: Image(
                            image: buildImageProvider(post.imageUrl),
                            fit: BoxFit.cover,
                          ),
                        );
                      },
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _openSettingsSheet(BuildContext context) {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (_) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.person_outline),
                title: const Text('Editar perfil'),
                onTap: () async {
                  Navigator.pop(context);
                  if (currentUser == null) return;

                  final updated = await Navigator.push<bool>(
                    context,
                    MaterialPageRoute(
                      builder: (_) =>
                          EditProfileScreen(currentUser: currentUser!),
                    ),
                  );

                  if (updated == true && onProfileUpdated != null) {
                    await onProfileUpdated!.call();
                  }
                },
              ),
              ListTile(
                leading: const Icon(Icons.logout, color: Colors.red),
                title: const Text(
                  'Cerrar sesión',
                  style: TextStyle(color: Colors.red),
                ),
                onTap: () async {
                  Navigator.pop(context);
                  if (onLogout != null) await onLogout!.call();
                },
              ),
            ],
          ),
        );
      },
    );
  }
}

class _ProfileStat extends StatelessWidget {
  final String label;
  final String value;

  const _ProfileStat({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 16,
            color: kUpsDarkBlue,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          style: const TextStyle(
            fontSize: 11,
            color: Colors.grey,
          ),
        ),
      ],
    );
  }
}
