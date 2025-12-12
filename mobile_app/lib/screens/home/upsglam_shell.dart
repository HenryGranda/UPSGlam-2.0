// lib/screens/home/upsglam_shell.dart
import 'package:flutter/material.dart';

import '../../models/current_user.dart';
import '../../models/post_model.dart';
import '../../models/comment_model.dart';

import '../../services/auth_service.dart';
import '../../services/post_service.dart';
import 'home_feed_view.dart';
import 'explore_view.dart';
import 'create_post_view.dart';
import 'notifications_view.dart';
import 'profile_view.dart';
import 'post_detail_page.dart';

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
      Map<String, dynamic>? data;

      // Siempre intentamos refrescar desde backend para obtener contadores al día
      data = await AuthService.instance.fetchCurrentUser();

      // Fallback a cache local si el llamado remoto falla o devuelve null
      data ??= await AuthService.instance.getStoredUser();

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

  Future<void> _refreshAll() async {
    await _loadCurrentUser();
    await _loadFeed();
  }

  Future<void> _loadFeed() async {
    try {
      final rawPosts = await PostService.instance.fetchFeed();
      if (!mounted) return;

      setState(() {
        _posts = rawPosts.map((json) {
          final post = PostModel.fromJson(json);

          // Si el post es tuyo y tienes photoUrl, usamos tu avatar actual
          if (_currentUser != null) {
            final cleanUsername = post.username.startsWith('@')
                ? post.username.substring(1)
                : post.username;

            if (cleanUsername == _currentUser!.username &&
                (_currentUser!.photoUrl != null &&
                    _currentUser!.photoUrl!.isNotEmpty)) {
              post.avatar = _currentUser!.photoUrl!;
            }
          }

          return post;
        }).toList();
      });
    } catch (e) {
      debugPrint('Error cargando feed: $e');
    }
  }

  Future<void> _toggleLike(String postId) async {
    final post = _posts.firstWhere((p) => p.id == postId);
    final wasLiked = post.liked; // estado ANTES del toggle

    // Actualización optimista -> que se vea al toque en la UI
    setState(() {
      post.liked = !post.liked;
      post.likes += wasLiked ? -1 : 1; // usamos wasLiked, no post.liked
      if (post.likes < 0) post.likes = 0; // por si ya tenías basura en BD
    });

    try {
      await PostService.instance.toggleLike(
        postId: postId,
        currentlyLiked: wasLiked,
      );
    } catch (e) {
      debugPrint('Error al hacer like/unlike: $e');

      // rollback si el backend falló
      setState(() {
        post.liked = wasLiked;
        post.likes += wasLiked ? 1 : -1;
        if (post.likes < 0) post.likes = 0;
      });
    }
  }

  void _incrementComments(String postId, int delta) {
    setState(() {
      final idx = _posts.indexWhere((p) => p.id == postId);
      if (idx != -1) {
        _posts[idx].commentsCount =
            (_posts[idx].commentsCount + delta).clamp(0, 999999);
      }
    });
  }

  // Compatibilidad con PostDetailPage (suma 1)
  void _addComment(String postId, CommentModel comment) {
    _incrementComments(postId, 1);
  }

  void _updateCaption(String postId, String newCaption) {
    setState(() {
      final idx = _posts.indexWhere((p) => p.id == postId);
      if (idx != -1) {
        _posts[idx].caption = newCaption;
      }
    });
  }

  void _removePost(String postId) {
    setState(() {
      _posts.removeWhere((p) => p.id == postId);
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
                onCaptionUpdated: _updateCaption,
                onDeletePost: _removePost,
                viewerUsername: _currentUser?.username,
              ),
            ),
          );
        },
        onToggleLike: _toggleLike,
        onRefresh: _refreshAll,
        onCommentsChanged: _incrementComments,
        onDeletePost: _removePost,
      ),
      ExploreView(
        posts: _posts,
        onRefresh: _refreshAll,
      ),
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
        onRefresh: _refreshAll,
        onDeletePost: _removePost,
        onCaptionUpdated: _updateCaption,
        viewerUsername: _currentUser?.username,
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
