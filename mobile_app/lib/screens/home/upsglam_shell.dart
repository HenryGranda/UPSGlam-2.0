import 'package:flutter/material.dart';

/// =============================
/// MODELOS
/// =============================

class CommentModel {
  final String id;
  final String username;
  final String avatar; // ruta asset o url
  final String text;
  final String timestamp;

  CommentModel({
    required this.id,
    required this.username,
    required this.avatar,
    required this.text,
    required this.timestamp,
  });
}

class PostModel {
  final String id;
  final String username;
  final String avatar;
  final String timestamp;
  final String imageUrl;
  final String caption;
  final String? filter;
  int likes;
  bool liked;
  final List<CommentModel> comments;

  PostModel({
    required this.id,
    required this.username,
    required this.avatar,
    required this.timestamp,
    required this.imageUrl,
    required this.caption,
    this.filter,
    this.likes = 0,
    this.liked = false,
    List<CommentModel>? comments,
  }) : comments = comments ?? [];
}

/// Datos de prueba (igual que en React)
final List<PostModel> dummyPosts = [
  PostModel(
    id: '1',
    username: '@maria_ups',
    avatar: 'assets/images/diverse_woman_avatar.png',
    timestamp: 'Hace 2 horas',
    imageUrl: 'assets/images/campus_university.jpg',
    caption: '¡Hermoso día en el campus UPS! 📚✨',
    likes: 42,
    filter: 'Filtro 1',
    comments: [
      CommentModel(
        id: 'c1',
        username: '@juan_tech',
        avatar: 'assets/images/man_avatar.png',
        text: '¡Me encanta ese filtro!',
        timestamp: 'Hace 1 hora',
      ),
    ],
  ),
  PostModel(
    id: '2',
    username: '@carlos_dev',
    avatar: 'assets/images/man_student.png',
    timestamp: 'Hace 5 horas',
    imageUrl: 'assets/images/coding_computer.jpg',
    caption: 'Trabajando en mi proyecto final 💻🔥',
    likes: 67,
    filter: 'Filtro 3',
  ),
  PostModel(
    id: '3',
    username: '@ana_design',
    avatar: 'assets/images/woman_designer.png',
    timestamp: 'Hace 1 día',
    imageUrl: 'assets/images/design_mockup.jpg',
    caption: 'Nuevo diseño para UPSGlam 🎨',
    likes: 89,
    filter: 'Filtro 2',
  ),
];

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
  late List<PostModel> _posts;

  @override
  void initState() {
    super.initState();
    _posts = dummyPosts
        .map(
          (p) => PostModel(
            id: p.id,
            username: p.username,
            avatar: p.avatar,
            timestamp: p.timestamp,
            imageUrl: p.imageUrl,
            caption: p.caption,
            likes: p.likes,
            liked: p.liked,
            filter: p.filter,
            comments: List<CommentModel>.from(p.comments),
          ),
        )
        .toList();
  }

  void _toggleLike(String postId) {
    setState(() {
      final post = _posts.firstWhere((p) => p.id == postId);
      post.liked = !post.liked;
      post.likes += post.liked ? 1 : -1;
    });
  }

  void _addPost(PostModel post) {
    setState(() {
      _posts.insert(0, post);
      _currentIndex = 0; // volvemos al home
    });
  }

  void _addComment(String postId, CommentModel comment) {
    setState(() {
      final post = _posts.firstWhere((p) => p.id == postId);
      post.comments.add(comment);
    });
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      HomeFeedView(
        posts: _posts,
        onOpenPost: (post) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => PostDetailPage(
                post: post,
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
        onPublish: _addPost,
        onBack: () => setState(() => _currentIndex = 0),
      ),
      const NotificationsView(),
      ProfileView(
        posts: _posts,
        onLogout: widget.onLogout,
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
  final void Function(PostModel) onOpenPost;
  final void Function(String postId) onToggleLike;

  const HomeFeedView({
    super.key,
    required this.posts,
    required this.onOpenPost,
    required this.onToggleLike,
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
          child: ListView.builder(
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
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            child: Row(
              children: [
                CircleAvatar(
                  radius: 20,
                  backgroundImage: AssetImage(post.avatar),
                  onBackgroundImageError: (_, __) {},
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        post.username,
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
              child: Image.asset(
                post.imageUrl,
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
                          const Icon(Icons.mode_comment_outlined, size: 24),
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
                        text: post.username,
                        style: const TextStyle(fontWeight: FontWeight.w600),
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

  const CreatePostView({
    super.key,
    required this.onPublish,
    required this.onBack,
  });

  @override
  State<CreatePostView> createState() => _CreatePostViewState();
}

class _CreatePostViewState extends State<CreatePostView> {
  final TextEditingController _usernameController =
      TextEditingController(text: '@usuario_ups');
  final TextEditingController _songController = TextEditingController();
  final TextEditingController _descriptionController =
      TextEditingController();

  // Imagen temporal antes de mandar al backend
  String _previewImage = 'assets/images/upload_image.jpg';

  // Lista fija de filtros que aplicará el backend CUDA
  final List<String> _filters = [
    'Filtro 1',
    'Filtro 2',
    'Filtro 3',
    'Filtro 4',
    'Filtro 5',
    'Filtro 6',
  ];

  String? _selectedFilter;

  /// VALIDA si se puede publicar
  bool get _canPublish =>
      _previewImage.isNotEmpty &&
      _previewImage != 'assets/images/upload_image.jpg' &&
      _descriptionController.text.trim().isNotEmpty;

  /// TODO por ahora
  void _onTakePhoto() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text("Función de cámara se conectará más adelante 📸"),
      ),
    );
  }

  /// TODO por ahora (más adelante usamos ImagePicker o backend)
  void _onPickFromGallery() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text("Galería se implementará más adelante 🖼️"),
      ),
    );
  }

  void _publish() {
    final post = PostModel(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      username: _usernameController.text.trim(),
      avatar: 'assets/images/abstract_avatar.png',
      timestamp: 'Ahora',
      imageUrl: _previewImage,
      caption: _descriptionController.text.trim(),
      filter: _selectedFilter,
      likes: 0,
      liked: false,
    );

    widget.onPublish(post);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("Publicación creada correctamente 🎉")),
    );

    _descriptionController.clear();
    _songController.clear();
    setState(() {
      _previewImage = 'assets/images/upload_image.jpg';
      _selectedFilter = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return GestureDetector(
      // SWIPE PARA SALIR (como Instagram)
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
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
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
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
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
                      child: Image.asset(
                        _previewImage,
                        fit: BoxFit.cover,
                        errorBuilder: (_, __, ___) =>
                            const Center(child: Icon(Icons.image)),
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
                          icon: const Icon(Icons.camera_alt_outlined),
                          label: const Text('Tomar foto'),
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 14),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _onPickFromGallery,
                          icon: const Icon(Icons.photo_library_outlined),
                          label: const Text('Galería'),
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 14),
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
                    children: _filters.map((name) {
                      final selected = _selectedFilter == name;
                      return ChoiceChip(
                        label: Text(name),
                        selected: selected,
                        onSelected: (_) {
                          setState(() {
                            _selectedFilter = selected ? null : name;
                          });
                        },
                        selectedColor: theme.colorScheme.primary,
                        labelStyle: TextStyle(
                          color: selected
                              ? theme.colorScheme.onPrimary
                              : theme.colorScheme.onSurface,
                        ),
                      );
                    }).toList(),
                  ),

                  const SizedBox(height: 20),

                  // CANCIÓN
                  Text(
                    'Agregar canción (opcional)',
                    style: theme.textTheme.titleSmall!
                        .copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 6),
                  TextField(
                    controller: _songController,
                    decoration: InputDecoration(
                      prefixIcon: const Icon(Icons.music_note_outlined),
                      hintText: 'Nombre de la canción',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
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
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // BOTÓN PUBLICAR
          Padding(
            padding:
                const EdgeInsets.only(left: 16, right: 16, bottom: 16, top: 4),
            child: SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: _canPublish ? _publish : null,
                child: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 14),
                  child: Text(
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
/// DETALLE DE PUBLICACIÓN (ACTUALIZADO)
/// =============================

class PostDetailPage extends StatefulWidget {
  final PostModel post;
  final void Function(String postId) onToggleLike;
  final void Function(String postId, CommentModel comment) onAddComment;

  const PostDetailPage({
    super.key,
    required this.post,
    required this.onToggleLike,
    required this.onAddComment,
  });

  @override
  State<PostDetailPage> createState() => _PostDetailPageState();
}

class _PostDetailPageState extends State<PostDetailPage> {
  late PostModel _post;
  final TextEditingController _commentController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _post = widget.post;
  }

  void _toggleLike() {
    widget.onToggleLike(_post.id);
    setState(() {
      _post.liked = !_post.liked;
      _post.likes += _post.liked ? 1 : -1;
    });
  }

  void _addComment() {
    final text = _commentController.text.trim();
    if (text.isEmpty) return;

    final comment = CommentModel(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      username: '@current_user',
      avatar: 'assets/images/abstract_avatar.png',
      text: text,
      timestamp: 'Ahora',
    );
    widget.onAddComment(_post.id, comment);
    setState(() {
      _post.comments.add(comment);
    });
    _commentController.clear();
  }

  void _openCommentsSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) {
        return Padding(
          padding: EdgeInsets.only(
            bottom: MediaQuery.of(context).viewInsets.bottom,
            top: 8,
          ),
          child: SizedBox(
            height: MediaQuery.of(context).size.height * 0.75,
            child: Column(
              children: [
                // Handle
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

                // Lista de comentarios
                Expanded(
                  child: _post.comments.isEmpty
                      ? const Center(
                          child: Text(
                            'No hay comentarios aún',
                            style: TextStyle(fontSize: 13, color: Colors.grey),
                          ),
                        )
                      : ListView.builder(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 16, vertical: 8),
                          itemCount: _post.comments.length,
                          itemBuilder: (context, index) {
                            final c = _post.comments[index];
                            return Padding(
                              padding: const EdgeInsets.symmetric(vertical: 6),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  CircleAvatar(
                                    radius: 16,
                                    backgroundImage: AssetImage(c.avatar),
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
                                              style: const TextStyle(
                                                fontWeight: FontWeight.w600,
                                                fontSize: 13,
                                              ),
                                            ),
                                            const SizedBox(width: 6),
                                            Text(
                                              c.timestamp,
                                              style: const TextStyle(
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
                                ],
                              ),
                            );
                          },
                        ),
                ),

                // Caja para escribir comentario
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                  child: Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _commentController,
                          decoration: InputDecoration(
                            hintText: 'Añadir un comentario…',
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(20),
                            ),
                            contentPadding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 8,
                            ),
                          ),
                          onSubmitted: (_) => _addComment(),
                        ),
                      ),
                      const SizedBox(width: 8),
                      IconButton(
                        onPressed: _addComment,
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
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Publicación'),
        centerTitle: true,
      ),
      body: Column(
        children: [
          // Info usuario
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            child: Row(
              children: [
                CircleAvatar(
                  radius: 22,
                  backgroundImage: AssetImage(_post.avatar),
                ),
                const SizedBox(width: 8),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _post.username,
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
            child: Image.asset(
              _post.imageUrl,
              fit: BoxFit.cover,
            ),
          ),

          // Info abajo
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
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
                                : Icons.favorite_border_outlined,
                            color: _post.liked ? Colors.pink : Colors.black87,
                            size: 28,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            '${_post.likes}',
                            style: const TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w600,
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
                          const Icon(Icons.mode_comment_outlined, size: 26),
                          const SizedBox(width: 4),
                          Text(
                            '${_post.comments.length}',
                            style: const TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 16),
                    const Icon(Icons.send_outlined, size: 26),
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
                        text: _post.username,
                        style: const TextStyle(fontWeight: FontWeight.w600),
                      ),
                      const TextSpan(text: ' '),
                      TextSpan(
                        text: _post.caption,
                        style: const TextStyle(color: Colors.black87),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 6),

                // Filtro aplicado (solo nombre, sin efecto visual)
                if (_post.filter != null)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.purple.shade50,
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Text(
                      'Filtro aplicado: ${_post.filter}',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: Colors.purple.shade700,
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
      return p.username.toLowerCase().contains(q) ||
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
                        child: Image.asset(
                          post.imageUrl,
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
  final Future<void> Function()? onLogout;

  const ProfileView({
    super.key,
    required this.posts,
    this.onLogout,
  });

  @override
  Widget build(BuildContext context) {
    final userPosts = posts.take(6).toList();

    return Column(
      children: [
        // Header con título + engranaje
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
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              IconButton(
                onPressed: () => _openSettingsSheet(context),
                icon: const Icon(Icons.settings_outlined),
              ),
            ],
          ),
        ),
        Expanded(
          child: SingleChildScrollView(
            child: Column(
              children: [
                const SizedBox(height: 20),
                CircleAvatar(
                  radius: 40,
                  backgroundImage:
                      const AssetImage('assets/images/user_profile.png'),
                  onBackgroundImageError: (_, __) {},
                ),
                const SizedBox(height: 12),
                const Text(
                  '@usuario_ups',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 32),
                  child: Text(
                    'Estudiante de UPS 🎓 | Amante de la fotografía 📸',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 13, color: Colors.grey),
                  ),
                ),
                const SizedBox(height: 16),
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
                Container(
                  decoration: const BoxDecoration(
                    border: Border(
                      top: BorderSide(color: Color(0xFFE5E7EB)),
                      bottom: BorderSide(color: Color(0xFFE5E7EB)),
                    ),
                  ),
                  child: Row(
                    children: const [
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
                  itemBuilder: (context, index) {
                    final post = userPosts[index];
                    return AspectRatio(
                      aspectRatio: 1,
                      child: Image.asset(
                        post.imageUrl,
                        fit: BoxFit.cover,
                        errorBuilder: (_, __, ___) => Container(
                          color: Colors.grey.shade300,
                          child: const Icon(Icons.image, size: 30),
                        ),
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
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 12),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 40,
                  height: 4,
                  margin: const EdgeInsets.only(bottom: 8),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(999),
                  ),
                ),
                const ListTile(
                  leading: Icon(Icons.person_outline),
                  title: Text('Editar perfil (próximamente)'),
                ),
                const Divider(height: 8),
                ListTile(
                  leading: const Icon(Icons.logout, color: Colors.red),
                  title: const Text(
                    'Cerrar sesión',
                    style: TextStyle(color: Colors.red),
                  ),
                  onTap: () async {
                    Navigator.pop(context); // cerrar sheet
                    if (onLogout != null) {
                      await onLogout!.call();
                    }
                  },
                ),
              ],
            ),
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
            fontSize: 18,
            fontWeight: FontWeight.bold,
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
