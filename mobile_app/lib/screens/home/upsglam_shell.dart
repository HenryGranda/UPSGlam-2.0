  import 'dart:io';

  import 'package:flutter/material.dart';
  import 'package:image_picker/image_picker.dart';

  import '../../screens/home/edit_profile_screen.dart';
  import '../../services/auth_service.dart';
  import '../../services/post_service.dart';
  import 'package:mobile_app/models/comment_model.dart';
  import 'package:mobile_app/models/current_user.dart';
  import 'package:mobile_app/models/post_model.dart';

  /// Helper para normalizar @username
  String formatUsername(String? raw) {
    if (raw == null || raw.isEmpty) return '@usuario_ups';
    return raw.startsWith('@') ? raw : '@$raw';
  }

  /// Construye un ImageProvider según el path/url
  ImageProvider _buildImageProvider(String path) {
    if (path.startsWith('http')) {
      return NetworkImage(path);
    }
    if (path.startsWith('assets/')) {
      return AssetImage(path);
    }
    return FileImage(File(path));
  }

  /// =============================
  /// SHELL PRINCIPAL CON BOTTOM BAR
  /// =============================

  class UPSGlamShell extends StatefulWidget {
    final Future<void> Function()? onLogout;

    const UPSGlamShell({super.key, this.onLogout});

    @override
    State<UPSGlamShell> createState() => _UPSGlamShellState();
  }

  class _UPSGlamShellState extends State<UPSGlamShell> {
    int _currentIndex = 0;
    List<PostModel> _posts = [];

    CurrentUser? _currentUser;
    bool _loadingUser = true;

    @override
    void initState() {
      super.initState();
      _initShell();
    }

    Future<void> _initShell() async {
      await _loadCurrentUser();
      await _loadFeed();
    }

    Future<void> _loadCurrentUser() async {
      try {
        Map<String, dynamic>? data =
            await AuthService.instance.getStoredUser();

        if (data == null) {
          data = await AuthService.instance.fetchCurrentUser();
        }

        if (!mounted) return;
        setState(() {
          if (data != null) {
            _currentUser = CurrentUser.fromJson(data);
          }
          _loadingUser = false;
        });
      } catch (_) {
        if (!mounted) return;
        setState(() {
          _loadingUser = false;
        });
      }
    }

    Future<void> _loadFeed() async {
      try {
        final rawPosts = await PostService.instance.fetchFeed();
        if (!mounted) return;
        setState(() {
          _posts = rawPosts.map((json) => PostModel.fromJson(json)).toList();
        });
      } catch (e) {
        debugPrint('Error cargando feed: $e');
      }
    }

    Future<void> _toggleLike(String postId) async {
      final post = _posts.firstWhere((p) => p.id == postId);
      final wasLiked = post.liked;

      // Actualización optimista
      setState(() {
        post.liked = !post.liked;
        post.likes += post.liked ? 1 : -1;
      });

      try {
        await PostService.instance.toggleLike(
          postId: postId,
          currentlyLiked: wasLiked,
        );
      } catch (e) {
        debugPrint('Error al hacer like/unlike: $e');

        // rollback si falló
        setState(() {
          post.liked = wasLiked;
          post.likes += post.liked ? 1 : -1;
        });
      }
    }

    // En UPSGlamShell (o donde tengas onAddComment)
    void _addComment(String postId, CommentModel comment) {
      setState(() {
        final idx = _posts.indexWhere((p) => p.id == postId);
        if (idx != -1) {
          // opcional: si tienes un campo separado de contador:
          // _posts[idx].commentsCount =
          //    (_posts[idx].commentsCount ?? 0) + 1;
        }
      });
    }

    @override
    Widget build(BuildContext context) {
      if (_loadingUser) {
        return const Scaffold(
          body: Center(child: CircularProgressIndicator()),
        );
      }

      final pages = [
        HomeFeedView(
          posts: _posts,
          currentUser: _currentUser,
          onOpenPost: (post) {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (_) => PostDetailPage(
                  post: post,
                  currentUser: _currentUser,
                  onToggleLike: _toggleLike,
                  onAddComment: _addComment,
                ),
              ),
            );
          },
          onToggleLike: _toggleLike,
        ),
        ExploreView(posts: _posts),
        CreatePostView(
          currentUser: _currentUser,
          onPublish: (_) {}, // no la usamos, recargamos el feed al volver
          onBack: () {
            setState(() => _currentIndex = 0);
            _loadFeed(); // recarga desde backend
          },
        ),
        const NotificationsView(),
        ProfileView(
          posts: _posts,
          currentUser: _currentUser,
          onLogout: widget.onLogout,
          onProfileUpdated: _loadCurrentUser,
        ),
      ];

      return Scaffold(
        backgroundColor: const Color(0xFFF5F7FB),
        body: SafeArea(
          child: pages[_currentIndex],
        ),
        bottomNavigationBar: BottomNavigationBar(
          currentIndex: _currentIndex,
          type: BottomNavigationBarType.fixed,
          selectedItemColor: Theme.of(context).colorScheme.primary,
          unselectedItemColor: Colors.grey,
          showSelectedLabels: false,
          showUnselectedLabels: false,
          onTap: (index) => setState(() => _currentIndex = index),
          items: const [
            BottomNavigationBarItem(
                icon: Icon(Icons.home_outlined), label: 'Inicio'),
            BottomNavigationBarItem(
                icon: Icon(Icons.search_outlined), label: 'Buscar'),
            BottomNavigationBarItem(
                icon: Icon(Icons.add_circle_outline), label: 'Crear'),
            BottomNavigationBarItem(
                icon: Icon(Icons.notifications_none), label: 'Notif'),
            BottomNavigationBarItem(
                icon: Icon(Icons.person_outline), label: 'Perfil'),
          ],
        ),
      );
    }
  }

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

  /// =============================
  /// HOME FEED (LISTA DE POSTS)
  /// =============================

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
                    Text(
                      'UPSGlam',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Theme.of(context).colorScheme.primary,
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
                  icon: const Icon(Icons.notifications_none),
                ),
              ],
            ),
          ),
          Expanded(
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
                              style:
                                  TextStyle(fontSize: 12, color: Colors.grey),
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
        shape: const Border(
          bottom: BorderSide(color: Color(0xFFE5E7EB)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // header usuario
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                children: [
                  CircleAvatar(
                    radius: 20,
                    backgroundImage: _buildImageProvider(post.avatar),
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
                          ),
                        ),
                        Text(
                          post.timestamp,
                          style: const TextStyle(
                            fontSize: 11,
                            color: Colors.grey,
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
                  image: _buildImageProvider(post.imageUrl),
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
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
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
                              color: post.liked ? Colors.pink : Colors.black87,
                              size: 26,
                            ),
                            const SizedBox(width: 4),
                            Text(
                              '${post.likes}',
                              style: const TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w500,
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
                            const Icon(Icons.mode_comment_outlined,
                                size: 24),
                            const SizedBox(width: 4),
                            Text(
                              '${post.comments.length}',
                              style: const TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 16),
                      const Icon(Icons.send_outlined, size: 24),
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
                          style:
                              const TextStyle(fontWeight: FontWeight.w600),
                        ),
                        const TextSpan(text: ' '),
                        TextSpan(
                          text: post.caption,
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
      );
    }
  }

  /// =============================
  /// CREAR PUBLICACIÓN 
  /// =============================

  class CreatePostView extends StatefulWidget {
    final void Function(PostModel) onPublish;
    final VoidCallback onBack;
    final CurrentUser? currentUser;

    const CreatePostView({
      super.key,
      required this.onPublish,
      required this.onBack,
      this.currentUser,
    });

    @override
    State<CreatePostView> createState() => _CreatePostViewState();
  }

  class _CreatePostViewState extends State<CreatePostView> {
    final TextEditingController _descriptionController =
        TextEditingController();
    final ImagePicker _picker = ImagePicker();

    String? _previewImagePath; // archivo local elegido
    bool _publishing = false;

    /// Filtros tal cual los espera el backend
    final List<String> _filters = [
      'gaussian',
      'box_blur',
      'prewitt',
      'laplacian',
      'ups_logo',
      'ups_color',
    ];

    String? _selectedFilter; // este string va directo al backend

    bool get _hasRealImage => _previewImagePath != null;
    bool get _canPublish =>
        _hasRealImage && _descriptionController.text.trim().isNotEmpty;

    // --- Cámara ---
    Future<void> _onTakePhoto() async {
      final picked =
          await _picker.pickImage(source: ImageSource.camera, imageQuality: 80);
      if (picked == null) return;
      setState(() {
        _previewImagePath = picked.path;
      });
    }

    // --- Galería ---
    Future<void> _onPickFromGallery() async {
      final picked =
          await _picker.pickImage(source: ImageSource.gallery, imageQuality: 80);
      if (picked == null) return;
      setState(() {
        _previewImagePath = picked.path;
      });
    }

    Future<void> _publish() async {
      if (!_canPublish || _previewImagePath == null) return;

      try {
        setState(() {
          _publishing = true;
        });

        final caption = _descriptionController.text.trim();
        final user = widget.currentUser;

        await PostService.instance.createPost(
          imageFile: File(_previewImagePath!),
          caption: caption,
          filter: _selectedFilter,
          username: user?.username,          
          userPhotoUrl: user?.photoUrl,     
        );

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Publicación creada correctamente")),
        );

        _descriptionController.clear();
        setState(() {
          _previewImagePath = null;
          _selectedFilter = null;
        });

        widget.onBack(); // vuelve al feed, donde recargas
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error al publicar: $e")),
        );
      } finally {
        setState(() {
          _publishing = false;
        });
      }
    }

    @override
    Widget build(BuildContext context) {
      final theme = Theme.of(context);

      return GestureDetector(
        // swipe hacia abajo para salir
        onVerticalDragEnd: (details) {
          if (details.primaryVelocity != null &&
              details.primaryVelocity! > 600) {
            widget.onBack();
          }
        },
        child: Column(
          children: [
            // HEADER
            SafeArea(
              bottom: false,
              child: Padding(
                padding: const EdgeInsets.symmetric(
                    horizontal: 12, vertical: 8),
                child: Row(
                  children: [
                    IconButton(
                      onPressed: widget.onBack,
                      icon: const Icon(Icons.arrow_back),
                    ),
                    const SizedBox(width: 8),
                    const Text(
                      'Crear publicación',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // CONTENIDO
            Expanded(
              child: SingleChildScrollView(
                padding:
                    const EdgeInsets.fromLTRB(16, 0, 16, 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // VISTA PREVIA
                    AspectRatio(
                      aspectRatio: 3 / 4,
                      child: Container(
                        decoration: BoxDecoration(
                          color: Colors.grey[200],
                          borderRadius: BorderRadius.circular(24),
                        ),
                        clipBehavior: Clip.antiAlias,
                        child: _previewImagePath == null
                            ? const Center(
                                child: Icon(Icons.image, size: 40),
                              )
                            : Image.file(
                                File(_previewImagePath!),
                                fit: BoxFit.cover,
                              ),
                      ),
                    ),

                    const SizedBox(height: 16),

                    // BOTONES: TOMAR FOTO / GALERÍA
                    Row(
                      children: [
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _onTakePhoto,
                            icon: const Icon(
                                Icons.camera_alt_outlined),
                            label: const Text('Tomar foto'),
                            style: OutlinedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(
                                  vertical: 14),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _onPickFromGallery,
                            icon: const Icon(
                                Icons.photo_library_outlined),
                            label: const Text('Galería'),
                            style: OutlinedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(
                                  vertical: 14),
                            ),
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 20),

                    // FILTROS
                    Text(
                      'Filtros GPU (backend)',
                      style: theme.textTheme.titleSmall!
                          .copyWith(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: _filters.map((code) {
                        final selected = _selectedFilter == code;
                        return ChoiceChip(
                          label: Text(code),
                          selected: selected,
                          onSelected: (_) {
                            setState(() {
                              _selectedFilter =
                                  selected ? null : code;
                            });
                          },
                          selectedColor:
                              theme.colorScheme.primary,
                          labelStyle: TextStyle(
                            color: selected
                                ? theme.colorScheme.onPrimary
                                : theme.colorScheme.onSurface,
                          ),
                        );
                      }).toList(),
                    ),

                    const SizedBox(height: 20),

                    // DESCRIPCIÓN
                    Text(
                      'Descripción',
                      style: theme.textTheme.titleSmall!
                          .copyWith(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 6),
                    TextField(
                      controller: _descriptionController,
                      maxLines: 3,
                      decoration: InputDecoration(
                        hintText: 'Escribe una descripción…',
                        border: OutlineInputBorder(
                          borderRadius:
                              BorderRadius.circular(16),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // BOTÓN PUBLICAR
            Padding(
              padding: const EdgeInsets.only(
                  left: 16, right: 16, bottom: 16, top: 4),
              child: SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed:
                      _canPublish && !_publishing ? _publish : null,
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                        vertical: 14),
                    child: _publishing
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Text(
                            'Publicar',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                  ),
                ),
              ),
            ),
          ],
        ),
      );
    }
  }

  /// =============================
  /// DETALLE DE PUBLICACIÓN 
  /// =============================

  class PostDetailPage extends StatefulWidget {
    final PostModel post;
    final CurrentUser? currentUser;
    final void Function(String postId) onToggleLike;
    final void Function(String postId, CommentModel comment) onAddComment;

    const PostDetailPage({
      super.key,
      required this.post,
      required this.onToggleLike,
      required this.onAddComment,
      this.currentUser,
    });

    @override
    State<PostDetailPage> createState() => _PostDetailPageState();
  }

  class _PostDetailPageState extends State<PostDetailPage> {
    late PostModel _post;
    final TextEditingController _commentController =
        TextEditingController();

    Future<void> _loadCommentsFromBackend() async {
      try {
        final comments =
            await PostService.instance.fetchComments(_post.id);

        if (!mounted) return;

        setState(() {
          _post.comments = comments;
        });
      } catch (e) {
        debugPrint('Error cargando comentarios del backend: $e');
      }
    }

    @override
    void initState() {
      super.initState();
      _post = widget.post; 
      _loadCommentsFromBackend(); 
    }

    void _toggleLike() {
      setState(() {
        if (_post.liked) {
          _post.liked = false;
          if (_post.likes > 0) {
            _post.likes -= 1;
          }
        } else {
          _post.liked = true;
          _post.likes += 1;
        }
      });

      try {
        widget.onToggleLike(_post.id);
      } catch (e) {
      }
    }

    void _openCommentsSheet() {
      showModalBottomSheet(
        context: context,
        isScrollControlled: true,
        shape: const RoundedRectangleBorder(
          borderRadius:
              BorderRadius.vertical(top: Radius.circular(24)),
        ),
        builder: (context) {
          // StatefulBuilder permite hacer setState SOLO dentro del modal
          return StatefulBuilder(
            builder: (context, modalSetState) {
              // EDITAR comentario (solo front)
              Future<void> editCommentAt(int index) async {
                final original = _post.comments[index];
                final controller = TextEditingController(text: original.text);

                final edited = await showDialog<String>(
                  context: context,
                  builder: (ctx) => AlertDialog(
                    title: const Text('Editar comentario'),
                    content: TextField(
                      controller: controller,
                      maxLines: 3,
                      decoration: const InputDecoration(
                        hintText: 'Edita tu comentario…',
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

                if (edited == null || edited.isEmpty || edited == original.text) return;

                try {
                  // 1) actualizar en backend
                  final updated = await PostService.instance.updateComment(
                    postId: _post.id,
                    commentId: original.id,
                    newText: edited,
                  );

                  // 2) actualizar en memoria
                  setState(() {
                    _post.comments[index] = updated;
                  });
                  modalSetState(() {});
                } catch (e) {
                  debugPrint('Error editando comentario: $e');
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('No se pudo editar el comentario')),
                  );
                }
              }

              // ELIMINAR comentario (solo front)
              Future<void> deleteCommentAt(int index) async {
                final comment = _post.comments[index];

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
                        child: const Text(
                          'Eliminar',
                          style: TextStyle(color: Colors.red),
                        ),
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

                  // si el backend dijo OK, actualizamos UI
                  setState(() {
                    _post.comments.removeAt(index);
                  });
                  modalSetState(() {});
                } catch (e) {
                  debugPrint('Error eliminando comentario: $e');
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('No se pudo eliminar el comentario'),
                    ),
                  );
                }
              }

              Future<void> addCommentFromSheet() async {
                final text = _commentController.text.trim();
                if (text.isEmpty) return;

                try {
                  // 1) Crear en backend y obtener el CommentModel "real"
                  final created = await PostService.instance.addComment(
                    postId: _post.id,
                    text: text,
                  );

                  // 2) Actualizar el detalle local
                  setState(() {
                    _post.comments.add(created);
                  });

                  // 3) Avisar al shell para que actualice el feed
                  widget.onAddComment(_post.id, created);

                  // 4) Refrescar el bottom sheet
                  modalSetState(() {});

                  // 5) Limpiar caja de texto
                  _commentController.clear();
                } catch (e) {
                  debugPrint('Error al crear comentario: $e');
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Error al publicar comentario')),
                  );
                }
              }

              return Padding(
                padding: EdgeInsets.only(
                  bottom: MediaQuery.of(context)
                      .viewInsets
                      .bottom,
                  top: 8,
                ),
                child: SizedBox(
                  height:
                      MediaQuery.of(context).size.height * 0.75,
                  child: Column(
                    children: [
                      // Handle
                      Container(
                        width: 40,
                        height: 4,
                        margin: const EdgeInsets.only(
                            top: 8, bottom: 12),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade300,
                          borderRadius:
                              BorderRadius.circular(999),
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

                      // Lista de comentarios
                      Expanded(
                        child: _post.comments.isEmpty
                            ? const Center(
                                child: Text(
                                  'No hay comentarios aún',
                                  style: TextStyle(
                                      fontSize: 13,
                                      color: Colors.grey),
                                ),
                              )
                            : ListView.builder(
                                padding:
                                    const EdgeInsets.symmetric(
                                        horizontal: 16,
                                        vertical: 8),
                                itemCount:
                                    _post.comments.length,
                                itemBuilder:
                                    (context, index) {
                                  final c =
                                      _post.comments[index];
                                  return Padding(
                                    padding:
                                        const EdgeInsets
                                                .symmetric(
                                            vertical: 6),
                                    child: Row(
                                      crossAxisAlignment:
                                          CrossAxisAlignment
                                              .start,
                                      children: [
                                        CircleAvatar(
                                          radius: 16,
                                          backgroundImage:
                                              _buildImageProvider(
                                                  c.avatar),
                                        ),
                                        const SizedBox(
                                            width: 8),
                                        Expanded(
                                          child: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment
                                                    .start,
                                            children: [
                                              Row(
                                                children: [
                                                  Text(
                                                    c.username,
                                                    style:
                                                        const TextStyle(
                                                      fontWeight:
                                                          FontWeight
                                                              .w600,
                                                      fontSize:
                                                          13,
                                                    ),
                                                  ),
                                                  const SizedBox(
                                                      width:
                                                          6),
                                                  Text(
                                                    c.timestamp,
                                                    style:
                                                        const TextStyle(
                                                      fontSize:
                                                          11,
                                                      color: Colors
                                                          .grey,
                                                    ),
                                                  ),
                                                ],
                                              ),
                                              const SizedBox(
                                                  height:
                                                      2),
                                              Text(
                                                c.text,
                                                style:
                                                    const TextStyle(
                                                  fontSize:
                                                      13,
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                        PopupMenuButton<
                                            String>(
                                          onSelected:
                                              (value) {
                                            if (value ==
                                                'edit') {
                                              editCommentAt(
                                                  index);
                                            } else if (value ==
                                                'delete') {
                                              deleteCommentAt(
                                                  index);
                                            }
                                          },
                                          itemBuilder:
                                              (context) =>
                                                  const [
                                            PopupMenuItem(
                                              value:
                                                  'edit',
                                              child: Text(
                                                  'Editar'),
                                            ),
                                            PopupMenuItem(
                                              value:
                                                  'delete',
                                              child: Text(
                                                  'Eliminar'),
                                            ),
                                          ],
                                        ),
                                      ],
                                    ),
                                  );
                                },
                              ),
                      ),

                      // Caja para escribir comentario
                      Padding(
                        padding:
                            const EdgeInsets.fromLTRB(
                                16, 8, 16, 16),
                        child: Row(
                          children: [
                            Expanded(
                              child: TextField(
                                controller:
                                    _commentController,
                                decoration:
                                    InputDecoration(
                                  hintText:
                                      'Añadir un comentario…',
                                  border:
                                      OutlineInputBorder(
                                    borderRadius:
                                        BorderRadius
                                            .circular(20),
                                  ),
                                  contentPadding:
                                      const EdgeInsets
                                          .symmetric(
                                    horizontal: 12,
                                    vertical: 8,
                                  ),
                                ),
                                onSubmitted: (_) =>
                                    addCommentFromSheet(),
                              ),
                            ),
                            const SizedBox(width: 8),
                            IconButton(
                              onPressed:
                                  addCommentFromSheet,
                              icon: const Icon(
                                  Icons.send),
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

    @override
    Widget build(BuildContext context) {
      final displayUsername = formatUsername(_post.username);

      return Scaffold(
        appBar: AppBar(
          title: const Text('Publicación'),
          centerTitle: true,
        ),
        body: Column(
          children: [
            // Info usuario
            Padding(
              padding: const EdgeInsets.symmetric(
                  horizontal: 16, vertical: 10),
              child: Row(
                children: [
                  CircleAvatar(
                    radius: 22,
                    backgroundImage:
                        _buildImageProvider(_post.avatar),
                  ),
                  const SizedBox(width: 8),
                  Column(
                    crossAxisAlignment:
                        CrossAxisAlignment.start,
                    children: [
                      Text(
                        displayUsername,
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 15,
                        ),
                      ),
                      Text(
                        _post.timestamp,
                        style: const TextStyle(
                          fontSize: 12,
                          color: Colors.grey,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // Imagen principal
            AspectRatio(
              aspectRatio: 1,
              child: Image(
                image:
                    _buildImageProvider(_post.imageUrl),
                fit: BoxFit.cover,
              ),
            ),

            // Info abajo
            Padding(
              padding: const EdgeInsets.symmetric(
                  horizontal: 16, vertical: 10),
              child: Column(
                crossAxisAlignment:
                    CrossAxisAlignment.start,
                children: [
                  // Botones like / comentarios / enviar
                  Row(
                    children: [
                      InkWell(
                        onTap: _toggleLike,
                        child: Row(
                          children: [
                            Icon(
                              _post.liked
                                  ? Icons.favorite
                                  : Icons
                                      .favorite_border_outlined,
                              color: _post.liked
                                  ? Colors.pink
                                  : Colors.black87,
                              size: 28,
                            ),
                            const SizedBox(width: 4),
                            Text(
                              '${_post.likes}',
                              style: const TextStyle(
                                fontSize: 15,
                                fontWeight:
                                    FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 16),
                      InkWell(
                        onTap: _openCommentsSheet,
                        child: Row(
                          children: [
                            const Icon(
                                Icons.mode_comment_outlined,
                                size: 26),
                            const SizedBox(width: 4),
                            Text(
                              '${_post.comments.length}',
                              style: const TextStyle(
                                fontSize: 15,
                                fontWeight:
                                    FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 16),
                      const Icon(Icons.send_outlined,
                          size: 26),
                    ],
                  ),

                  const SizedBox(height: 8),

                  // Nº me gusta
                  Text(
                    '${_post.likes} me gusta',
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                    ),
                  ),

                  const SizedBox(height: 6),

                  // Descripción
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
                              fontWeight: FontWeight.w600),
                        ),
                        const TextSpan(text: ' '),
                        TextSpan(
                          text: _post.caption,
                          style: const TextStyle(
                              color: Colors.black87),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 6),

                  // Filtro aplicado (solo nombre, sin efecto visual)
                  if (_post.filter != null)
                    Container(
                      padding:
                          const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.purple.shade50,
                        borderRadius:
                            BorderRadius.circular(999),
                      ),
                      child: Text(
                        'Filtro aplicado: ${_post.filter}',
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          color: Colors
                              .purple.shade700,
                        ),
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

  /// =============================
  /// EXPLORAR
  /// =============================

  class ExploreView extends StatefulWidget {
    final List<PostModel> posts;

    const ExploreView({super.key, required this.posts});

    @override
    State<ExploreView> createState() => _ExploreViewState();
  }

  class _ExploreViewState extends State<ExploreView> {
    String _search = '';

    @override
    Widget build(BuildContext context) {
      final filtered = widget.posts.where((p) {
        if (_search.trim().isEmpty) return true;
        final q = _search.toLowerCase();
        return formatUsername(p.username).toLowerCase().contains(q) ||
            p.caption.toLowerCase().contains(q);
      }).toList();

      return Column(
        children: [
          // Header + buscador
          Container(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            decoration: const BoxDecoration(
              color: Colors.white,
              border: Border(
                bottom: BorderSide(color: Color(0xFFE5E7EB)),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Explorar',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 8),
                TextField(
                  decoration: InputDecoration(
                    prefixIcon: const Icon(Icons.search),
                    hintText: 'Buscar usuarios o publicaciones…',
                    isDense: true,
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(24),
                    ),
                    contentPadding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  ),
                  onChanged: (value) {
                    setState(() => _search = value);
                  },
                ),
              ],
            ),
          ),

          // Grid de imágenes
          Expanded(
            child: filtered.isEmpty
                ? const Center(
                    child: Text(
                      'No se encontraron resultados',
                      style: TextStyle(color: Colors.grey),
                    ),
                  )
                : GridView.builder(
                    padding: const EdgeInsets.all(1),
                    gridDelegate:
                        const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 3,
                      mainAxisSpacing: 1,
                      crossAxisSpacing: 1,
                    ),
                    itemCount: filtered.length,
                    itemBuilder: (context, index) {
                      final post = filtered[index];
                      return GestureDetector(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => PostDetailPage(
                                post: post,
                                onToggleLike: (id) {},
                                onAddComment: (id, c) {},
                              ),
                            ),
                          );
                        },
                        child: AspectRatio(
                          aspectRatio: 1,
                          child: Image(
                            image: _buildImageProvider(post.imageUrl),
                            fit: BoxFit.cover,
                            errorBuilder: (_, __, ___) => Container(
                              color: Colors.grey.shade300,
                              child: const Icon(Icons.image, size: 30),
                            ),
                          ),
                        ),
                      );
                    },
                  ),
          ),
        ],
      );
    }
  }

  /// =============================
  /// NOTIFICACIONES
  /// =============================

  class NotificationsView extends StatelessWidget {
    const NotificationsView({super.key});

    @override
    Widget build(BuildContext context) {
      final notifications = <_NotificationItem>[
        _NotificationItem(
          type: NotificationType.like,
          title: '@maria_ups le dio like a tu publicación',
          time: 'Hace 2 min',
        ),
        _NotificationItem(
          type: NotificationType.comment,
          title: '@carlos_dev comentó: "Brutal ese filtro 🤯"',
          time: 'Hace 10 min',
        ),
        _NotificationItem(
          type: NotificationType.follow,
          title: '@ana_design empezó a seguirte',
          time: 'Hace 1 hora',
        ),
      ];

      return Column(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(vertical: 12),
            decoration: const BoxDecoration(
              color: Colors.white,
              border: Border(bottom: BorderSide(color: Color(0xFFE5E7EB))),
            ),
            child: const Center(
              child: Text(
                'Notificaciones',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          Expanded(
            child: ListView.separated(
              itemCount: notifications.length,
              separatorBuilder: (_, __) => const Divider(height: 0),
              itemBuilder: (context, index) {
                final n = notifications[index];
                return ListTile(
                  leading: CircleAvatar(
                    backgroundColor: _colorForType(n.type),
                    child: Icon(
                      _iconForType(n.type),
                      color: Colors.white,
                      size: 18,
                    ),
                  ),
                  title: Text(
                    n.title,
                    style: const TextStyle(fontSize: 13),
                  ),
                  subtitle: Text(
                    n.time,
                    style: const TextStyle(fontSize: 11, color: Colors.grey),
                  ),
                  onTap: () {
                    // Más adelante: navegar a la publicación o perfil
                  },
                );
              },
            ),
          ),
        ],
      );
    }

    static IconData _iconForType(NotificationType type) {
      switch (type) {
        case NotificationType.like:
          return Icons.favorite;
        case NotificationType.comment:
          return Icons.mode_comment_outlined;
        case NotificationType.follow:
          return Icons.person_add_alt_1;
      }
    }

    static Color _colorForType(NotificationType type) {
      switch (type) {
        case NotificationType.like:
          return Colors.pink;
        case NotificationType.comment:
          return Colors.blueAccent;
        case NotificationType.follow:
          return Colors.green;
      }
    }
  }

  enum NotificationType { like, comment, follow }

  class _NotificationItem {
    final NotificationType type;
    final String title;
    final String time;

    _NotificationItem({
      required this.type,
      required this.title,
      required this.time,
    });
  }

  /// =============================
  /// PERFIL
  /// =============================

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
      final ImageProvider<Object> avatar =
          (user?.photoUrl != null && user!.photoUrl!.isNotEmpty)
              ? NetworkImage(user.photoUrl!) as ImageProvider<Object>
              : const AssetImage('assets/images/user_profile.png')
                  as ImageProvider<Object>;

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
                          image: _buildImageProvider(post.imageUrl),
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
