// lib/screens/home/home_feed_view.dart
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:visibility_detector/visibility_detector.dart';

import '../../models/current_user.dart';
import '../../models/post_model.dart';
import '../../models/comment_model.dart';
import '../../services/post_service.dart';
import '../../services/audio_manager.dart';

import 'public_profile_page.dart';
import 'ui_helpers.dart';

/// Colores inspirados en el logo de la UPS
const kUpsDarkBlue = Color(0xFF003A70);
const kUpsLightBlue = Color(0xFF4F8EC7);
const kUpsYellow = Color(0xFFF4C430);

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
  final Future<void> Function()? onRefresh;
  final void Function(String postId, int deltaComments)? onCommentsChanged;
  final void Function(String postId)? onDeletePost;

  const HomeFeedView({
    super.key,
    required this.posts,
    required this.onOpenPost,
    required this.onToggleLike,
    this.currentUser,
    this.onRefresh,
    this.onCommentsChanged,
    this.onDeletePost,
  });

  @override
  Widget build(BuildContext context) {
    final refreshableList = ListView.builder(
      physics: const AlwaysScrollableScrollPhysics(),
      padding: EdgeInsets.zero,
      itemCount: posts.isEmpty ? 1 : posts.length + 1,
      itemBuilder: (context, index) {
        if (posts.isEmpty) {
          return const _PlaceholderCenter(text: 'Fin del contenido');
        }
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
          onCommentsChanged: onCommentsChanged,
          currentUser: currentUser,
          onDeletePost: onDeletePost,
        );
      },
    );

    return Column(
      children: [
        // AppBar custom
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          decoration: const BoxDecoration(
            color: Colors.white,
            border: Border(
              bottom: BorderSide(color: Color(0xFFE5E7EB)),
            ),
          ),
          child: Row(
            children: [
              Row(
                children: [
                  Image.asset(
                    'assets/images/logo_upsglam_small.png',
                    height: 28,
                    errorBuilder: (_, __, ___) =>
                        const Icon(Icons.camera_alt_outlined),
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    'UPSGlam',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: kUpsDarkBlue,
                    ),
                  ),
                ],
              ),
              const Spacer(),
              const Text(
                'Inicio',
                style: TextStyle(fontWeight: FontWeight.w600),
              ),
              const Spacer(),
              IconButton(
                onPressed: () {},
                icon: const Icon(Icons.notifications_none, color: kUpsDarkBlue),
              ),
            ],
          ),
        ),

        Expanded(
          child: RefreshIndicator(
            onRefresh: onRefresh ?? () async {},
            child: refreshableList,
          ),
        ),
      ],
    );
  }
}

/// Tarjeta individual de publicación
class PostCard extends StatefulWidget {
  final PostModel post;
  final VoidCallback onOpen;
  final VoidCallback onToggleLike;
  final void Function(String postId, int deltaComments)? onCommentsChanged;
  final CurrentUser? currentUser;
  final void Function(String postId)? onDeletePost;

  const PostCard({
    super.key,
    required this.post,
    required this.onOpen,
    required this.onToggleLike,
    this.onCommentsChanged,
    this.currentUser,
    this.onDeletePost,
  });

  @override
  State<PostCard> createState() => _PostCardState();
}

class _PostCardState extends State<PostCard> {
  late PostModel _post;
  AudioPlayer? _audioPlayer;
  bool _isMuted = false;
  final AudioManager _audioManager = AudioManager();

  @override
  void initState() {
    super.initState();
    _post = widget.post;
    _initAudio();
  }

  void _initAudio() {
    debugPrint('🎵 DEBUG: Post ${_post.id} audioFile = ${_post.audioFile}');
    if (_post.audioFile != null && _post.audioFile!.isNotEmpty) {
      debugPrint('🎵 Inicializando AudioPlayer para: ${_post.audioFile}');
      _audioPlayer = AudioPlayer();
      _audioPlayer!.setReleaseMode(ReleaseMode.loop);
      // NO reproducir automáticamente, solo cuando sea visible
    } else {
      debugPrint('⚠️ Post sin audioFile');
    }
  }

  @override
  void didUpdateWidget(covariant PostCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.post != widget.post) {
      _post = widget.post;
    }
  }

  @override
  void dispose() {
    // Pausar audio si este post se está destruyendo
    if (_audioPlayer != null) {
      _audioManager.pauseAudio(_post.id);
    }
    _audioPlayer?.dispose();
    super.dispose();
  }

  void _toggleMute() {
    if (_audioPlayer == null) return;
    setState(() {
      _isMuted = !_isMuted;
      _audioPlayer!.setVolume(_isMuted ? 0.0 : 1.0);
    });
  }

  /// Se llama cuando el post se vuelve visible o invisible
  void _onVisibilityChanged(VisibilityInfo info) {
    if (_audioPlayer == null || _post.audioFile == null) return;

    // Si es visible más del 50%, reproducir
    if (info.visibleFraction > 0.5) {
      debugPrint('👁️ Post ${_post.id} visible (${(info.visibleFraction * 100).toStringAsFixed(0)}%)');
      _audioManager.playAudio(
        postId: _post.id,
        player: _audioPlayer!,
        audioFile: _post.audioFile!,
      );
    } else {
      // Si es menos del 50% visible, pausar
      debugPrint('🙈 Post ${_post.id} no visible (${(info.visibleFraction * 100).toStringAsFixed(0)}%)');
      _audioManager.pauseAudio(_post.id);
    }
  }

  String _getAudioLabel() {
    if (_post.audioFile == null) return '';
    
    final audioLabels = {
      'Siuu.mp3': 'Siuuu',
      'cancioncr7_FRPRUlK5.mp3': 'Canción CR7',
      'vamonosamartecr7_p9FG2LGY.mp3': 'Vámonos a Amar',
      'waka_IEJmUo0a.mp3': 'Waka Waka',
      'kuduro_isTLScH6.mp3': 'Kuduro',
      'nadiedominar_XcY3jlnG.mp3': 'Nadie Dominar',
    };
    
    return audioLabels[_post.audioFile] ?? 'Audio';
  }

  @override
  Widget build(BuildContext context) {
    final displayUsername = formatUsername(_post.username);

    void _openUserProfile() {
      final cleanUsername =
          _post.username.startsWith('@') ? _post.username.substring(1) : _post.username;

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => PublicProfilePage(username: cleanUsername),
        ),
      );
    }

    return VisibilityDetector(
      key: Key('post_${_post.id}'),
      onVisibilityChanged: _onVisibilityChanged,
      child: Card(
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
                GestureDetector(
                  onTap: _openUserProfile,
                  child: CircleAvatar(
                    radius: 20,
                    backgroundColor: kUpsLightBlue.withOpacity(0.15),
                    backgroundImage: buildImageProvider(_post.avatar),
                    onBackgroundImageError: (_, __) {},
                  ),
                ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                      GestureDetector(
                        onTap: _openUserProfile,
                        child: Text(
                          displayUsername,
                          style: const TextStyle(
                            fontWeight: FontWeight.w600,
                            fontSize: 13,
                            color: kUpsDarkBlue,
                          ),
                        ),
                      ),
                        Text(
                          _post.timestamp,
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
            onTap: widget.onOpen,
            child: AspectRatio(
              aspectRatio: 4 / 5,
              child: Stack(
                children: [
                  Image(
                    image: buildImageProvider(_post.imageUrl),
                    fit: BoxFit.cover,
                    width: double.infinity,
                    height: double.infinity,
                    errorBuilder: (_, __, ___) => Container(
                      color: Colors.grey.shade300,
                      child: const Icon(Icons.image, size: 40),
                    ),
                  ),
                  // Botón de mute/unmute si tiene audio
                  if (_audioPlayer != null)
                    Positioned(
                      top: 12,
                      right: 12,
                      child: Container(
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.6),
                          shape: BoxShape.circle,
                        ),
                        child: IconButton(
                          icon: Icon(
                            _isMuted ? Icons.volume_off : Icons.volume_up,
                            color: Colors.white,
                            size: 22,
                          ),
                          onPressed: _toggleMute,
                          tooltip: _isMuted ? 'Activar audio' : 'Silenciar',
                        ),
                      ),
                    ),
                  // Indicador de música activa
                  if (_audioPlayer != null)
                    Positioned(
                      bottom: 12,
                      left: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.6),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              _isMuted ? Icons.music_off : Icons.music_note,
                              color: Colors.white,
                              size: 16,
                            ),
                            const SizedBox(width: 6),
                            Text(
                              _getAudioLabel(),
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                ],
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
                      onTap: widget.onToggleLike,
                      child: Row(
                        children: [
                          Icon(
                            _post.liked
                                ? Icons.favorite
                                : Icons.favorite_border_outlined,
                            color: _post.liked ? kUpsYellow : kUpsDarkBlue,
                            size: 26,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            '${_post.likes}',
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
                      onTap: () => _openCommentsSheet(context),
                      child: Row(
                        children: [
                          const Icon(Icons.mode_comment_outlined,
                              size: 24, color: kUpsDarkBlue),
                          const SizedBox(width: 4),
                          Text(
                            '${_post.commentsCount}',
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
                    const Icon(Icons.send_outlined,
                        size: 24, color: kUpsDarkBlue),
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
                        text: _post.caption,
                        style: const TextStyle(color: Colors.black54),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    ),
    );
  }

  bool get _isOwner {
    final current = widget.currentUser?.username;
    if (current == null) return false;
    final cleanCurrent = current.startsWith('@') ? current.substring(1) : current;
    final cleanPost = _post.username.startsWith('@') ? _post.username.substring(1) : _post.username;
    return cleanCurrent == cleanPost;
  }

  void _openCommentsSheet(BuildContext context) {
    final TextEditingController controller = TextEditingController();
    List<CommentModel> comments = [];
    bool loading = true;
    String? error;

    Future<void> loadComments(StateSetter setSheet) async {
      try {
        comments = await PostService.instance.fetchComments(_post.id);
        loading = false;
        error = null;
      } catch (e) {
        loading = false;
        error = 'No se pudieron cargar los comentarios';
      }
      setSheet(() {});
    }

    Future<void> addComment(StateSetter setSheet) async {
      final text = controller.text.trim();
      if (text.isEmpty) return;
      try {
        final created = await PostService.instance.addComment(
          postId: _post.id,
          text: text,
        );
        comments.add(created);
        controller.clear();
        widget.onCommentsChanged?.call(_post.id, 1);
        setSheet(() {});
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Error al publicar comentario')),
        );
      }
    }

    Future<void> editCommentAt(int index, StateSetter setSheet) async {
      final original = comments[index];
      final controllerEdit = TextEditingController(text: original.text);

      final edited = await showDialog<String>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Editar comentario'),
          content: TextField(
            controller: controllerEdit,
            maxLines: 3,
            decoration: const InputDecoration(
              hintText: 'Edita tu comentario',
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancelar'),
            ),
            TextButton(
              onPressed: () =>
                  Navigator.pop(ctx, controllerEdit.text.trim()),
              child: const Text('Guardar'),
            ),
          ],
        ),
      );

      if (edited == null || edited.isEmpty || edited == original.text) return;

      try {
        final updated = await PostService.instance.updateComment(
          postId: _post.id,
          commentId: original.id,
          newText: edited,
        );
        comments[index] = updated;
        setSheet(() {});
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No se pudo editar el comentario')),
        );
      }
    }

    Future<void> deleteCommentAt(int index, StateSetter setSheet) async {
      final comment = comments[index];

      final confirm = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Eliminar comentario'),
          content: const Text('¿Seguro que quieres eliminar este comentario?'),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancelar'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Eliminar', style: TextStyle(color: Colors.red)),
            ),
          ],
        ),
      );

      if (confirm != true) return;

      try {
        await PostService.instance.deleteComment(
          postId: _post.id,
          commentId: comment.id,
        );
        comments.removeAt(index);
        widget.onCommentsChanged?.call(_post.id, -1);
        setSheet(() {});
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No se pudo eliminar el comentario')),
        );
      }
    }

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setSheet) {
            if (loading) {
              loadComments(setSheet);
            }
            return Padding(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).viewInsets.bottom,
                top: 8,
              ),
              child: SizedBox(
                height: MediaQuery.of(context).size.height * 0.75,
                child: Column(
                  children: [
                    Container(
                      width: 40,
                      height: 4,
                      margin: const EdgeInsets.only(top: 8, bottom: 12),
                      decoration: BoxDecoration(
                        color: Colors.grey.shade300,
                        borderRadius: BorderRadius.circular(999),
                      ),
                    ),
                    const Text(
                      'Comentarios',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        fontSize: 16,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Expanded(
                      child: loading
                          ? const Center(child: CircularProgressIndicator())
                          : (error != null
                              ? Center(
                                  child: Text(
                                  error!,
                                  style: const TextStyle(color: Colors.grey),
                                ))
                              : (comments.isEmpty
                                  ? const Center(
                                      child: Text(
                                        'No hay comentarios aún',
                                        style: TextStyle(
                                            fontSize: 13, color: Colors.grey),
                                      ),
                                    )
                                  : ListView.builder(
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 16, vertical: 8),
                                      itemCount: comments.length,
                                      itemBuilder: (context, index) {
                                        final c = comments[index];
                                        final current = widget.currentUser?.username ?? '';
                                        final cleanCurrent = current.startsWith('@') ? current.substring(1) : current;
                                        final cleanComment = c.username.startsWith('@') ? c.username.substring(1) : c.username;
                                        final canEditDelete = cleanCurrent.isNotEmpty && cleanCurrent == cleanComment;

                                        return Padding(
                                          padding: const EdgeInsets.symmetric(
                                              vertical: 6),
                                          child: Row(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              CircleAvatar(
                                                radius: 16,
                                                backgroundImage:
                                                    buildImageProvider(
                                                        c.avatar),
                                              ),
                                              const SizedBox(width: 8),
                                              Expanded(
                                                child: Column(
                                                  crossAxisAlignment:
                                                      CrossAxisAlignment.start,
                                                  children: [
                                                    Row(
                                                      children: [
                                                        Text(
                                                          c.username,
                                                          style:
                                                              const TextStyle(
                                                            fontWeight:
                                                                FontWeight.w600,
                                                            fontSize: 13,
                                                          ),
                                                        ),
                                                        const SizedBox(width: 6),
                                                        Text(
                                                          c.timestamp,
                                                          style:
                                                              const TextStyle(
                                                            fontSize: 11,
                                                            color: Colors.grey,
                                                          ),
                                                        ),
                                                      ],
                                                    ),
                                                    const SizedBox(height: 2),
                                                    Text(
                                                      c.text,
                                                      style: const TextStyle(
                                                        fontSize: 13,
                                                      ),
                                                    ),
                                                  ],
                                                ),
                                              ),
                                              if (canEditDelete)
                                                PopupMenuButton<String>(
                                                  onSelected: (value) {
                                                    if (value == 'edit') {
                                                      editCommentAt(index, setSheet);
                                                    } else if (value == 'delete') {
                                                      deleteCommentAt(index, setSheet);
                                                    }
                                                  },
                                                  itemBuilder: (context) => const [
                                                    PopupMenuItem(
                                                      value: 'edit',
                                                      child: Text('Editar'),
                                                    ),
                                                    PopupMenuItem(
                                                      value: 'delete',
                                                      child: Text('Eliminar'),
                                                    ),
                                                  ],
                                                ),
                                            ],
                                          ),
                                        );
                                      },
                                    ))),
                    ),
                    Padding(
                      padding:
                          const EdgeInsets.fromLTRB(16, 8, 16, 16),
                      child: Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: controller,
                              decoration: InputDecoration(
                                hintText: 'Añadir un comentario',
                                border: OutlineInputBorder(
                                  borderRadius:
                                      BorderRadius.circular(20),
                                ),
                                contentPadding:
                                    const EdgeInsets.symmetric(
                                  horizontal: 12,
                                  vertical: 8,
                                ),
                              ),
                              onSubmitted: (_) => addComment(setSheet),
                            ),
                          ),
                          const SizedBox(width: 8),
                          IconButton(
                            onPressed: () => addComment(setSheet),
                            icon: const Icon(Icons.send),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  Future<void> _editPostCaption(BuildContext context) async {
    final controller = TextEditingController(text: _post.caption);
    final newText = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Editar publicación'),
        content: TextField(
          controller: controller,
          maxLines: 3,
          decoration: const InputDecoration(
            hintText: 'Nueva descripción',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancelar'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, controller.text.trim()),
            child: const Text('Guardar'),
          ),
        ],
      ),
    );

    if (newText == null || newText.isEmpty || newText == _post.caption) return;

    try {
      await PostService.instance.updatePostCaption(
        postId: _post.id,
        newCaption: newText,
      );
      setState(() {
        _post.caption = newText;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Publicación actualizada')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error al actualizar: $e')),
      );
    }
  }

  Future<void> _deletePost(BuildContext context) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Eliminar publicación'),
        content:
            const Text('¿Seguro que deseas eliminar esta publicación?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancelar'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Eliminar', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );

    if (confirm != true) return;

    try {
      await PostService.instance.deletePost(_post.id);
      widget.onDeletePost?.call(_post.id);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Publicación eliminada')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error al eliminar: $e')),
      );
    }
  }
}
