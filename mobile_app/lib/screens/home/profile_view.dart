// lib/screens/home/profile_view.dart
import 'package:flutter/material.dart';

import '../../models/current_user.dart';
import '../../models/post_model.dart';
import '../../models/avatars.dart';
import '../../screens/home/edit_profile_screen.dart';

import 'post_detail_page.dart';
import 'ui_helpers.dart';

class ProfileView extends StatelessWidget {
  final List<PostModel> posts;
  final CurrentUser? currentUser;
  final Future<void> Function()? onLogout;
  final Future<void> Function()? onProfileUpdated;

  const ProfileView({
    super.key,
    required this.posts,
    this.currentUser,
    this.onLogout,
    this.onProfileUpdated,
  });

  @override
  Widget build(BuildContext context) {
    final user = currentUser;

    // USERNAME DINÁMICO
    final username = formatUsername(user?.username);

    // BIO DINÁMICA
    final bio = (user?.bio != null && user!.bio!.trim().isNotEmpty)
        ? user.bio!
        : 'Añade una biografía para que te conozcan mejor ✍️';

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

    return Column(
      children: [
        // HEADER
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: const BoxDecoration(
            color: Colors.white,
            border: Border(bottom: BorderSide(color: Color(0xFFE5E7EB))),
          ),
          child: Row(
            children: [
              const Spacer(),
              const Text(
                'Perfil',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
              ),
              const Spacer(),
              IconButton(
                onPressed: () => _openSettingsSheet(context),
                icon: const Icon(Icons.settings_outlined),
              ),
            ],
          ),
        ),

        // CONTENIDO
        Expanded(
          child: SingleChildScrollView(
            child: Column(
              children: [
                const SizedBox(height: 20),

                // FOTO
                CircleAvatar(
                  radius: 40,
                  backgroundImage: avatar,
                ),

                const SizedBox(height: 12),

                // USERNAME
                Text(
                  username,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
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

                const SizedBox(height: 16),

                // ESTADÍSTICAS
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _ProfileStat(
                      label: 'Publicaciones',
                      value: '${userPosts.length}',
                    ),
                    const _ProfileStat(label: 'Seguidores', value: '342'),
                    const _ProfileStat(label: 'Siguiendo', value: '128'),
                  ],
                ),

                const SizedBox(height: 16),

                // TABS PUBLICACIONES / GUARDADAS
                Container(
                  decoration: const BoxDecoration(
                    border: Border(
                      top: BorderSide(color: Color(0xFFE5E7EB)),
                      bottom: BorderSide(color: Color(0xFFE5E7EB)),
                    ),
                  ),
                  child: const Row(
                    children: [
                      Expanded(
                        child: Padding(
                          padding: EdgeInsets.symmetric(vertical: 12),
                          child: Center(
                            child: Text(
                              'Publicaciones',
                              style: TextStyle(
                                fontWeight: FontWeight.w600,
                                color: Colors.purple,
                              ),
                            ),
                          ),
                        ),
                      ),
                      Expanded(
                        child: Padding(
                          padding: EdgeInsets.symmetric(vertical: 12),
                          child: Center(
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
      ],
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
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          style: const TextStyle(fontSize: 11, color: Colors.grey),
        ),
      ],
    );
  }
}
