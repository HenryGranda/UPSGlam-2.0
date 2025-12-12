import 'package:flutter/material.dart';

import '../../models/current_user.dart';
import '../../models/post_model.dart';
import '../../services/auth_service.dart';
import '../../services/post_service.dart';
import '../auth/login_screen.dart';
import 'profile_view.dart';

class PublicProfilePage extends StatefulWidget {
  final String username; // con o sin @

  const PublicProfilePage({
    super.key,
    required this.username,
  });

  @override
  State<PublicProfilePage> createState() => _PublicProfilePageState();
}

class _PublicProfilePageState extends State<PublicProfilePage> {
  CurrentUser? _user;
  List<PostModel> _posts = [];
  bool _loading = true;

  bool _isMe = false;
  bool _isFollowing = false;

  String get _cleanUsername {
    final u = widget.username.trim();
    return u.startsWith('@') ? u.substring(1) : u;
  }

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _forceToLogin() async {
    await AuthService.instance.logout();
    if (!mounted) return;
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(
        builder: (_) => LoginScreen(
          onLoginSuccess: () {
            // al volver a loguear, regresa al home y desde ahí navegas de nuevo
          },
        ),
      ),
      (_) => false,
    );
  }

  bool _looksLikeUnauthorized(Object e) {
    final msg = e.toString();
    return msg.contains('Error 401') ||
        msg.contains('UNAUTHORIZED') ||
        msg.contains('Token inválido') ||
        msg.contains('expired') ||
        msg.contains('expirado');
  }

  Future<void> _loadData() async {
    setState(() => _loading = true);

    try {
      // 1️⃣ Perfil público desde BACKEND
      final userJson =
          await AuthService.instance.fetchUserByUsername(_cleanUsername);

      final user = CurrentUser.fromJson(userJson);

      // flags calculados por backend
      final isMe = userJson['isMe'] == true;
      final isFollowing = userJson['isFollowing'] == true;

      // 2️⃣ Posts del usuario
      final rawPosts =
          await PostService.instance.fetchPostsByUsername(_cleanUsername);

      final posts =
          rawPosts.map<PostModel>((j) => PostModel.fromJson(j)).toList();

      if (!mounted) return;
      setState(() {
        _user = user;
        _posts = posts;
        _isMe = isMe;
        _isFollowing = isFollowing;
        _loading = false;
      });
    } catch (e) {
      debugPrint('Error cargando perfil público: $e');

      // Si es token expirado / 401 => logout y a login
      if (_looksLikeUnauthorized(e)) {
        await _forceToLogin();
        return;
      }

      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  Future<void> _toggleFollow() async {
    final target = _user;
    if (target == null) return;

    final wasFollowing = _isFollowing;

    // UI optimista
    setState(() {
      _isFollowing = !wasFollowing;
      final newFollowers =
          (target.followersCount + (wasFollowing ? -1 : 1)).clamp(0, 999999);
      _user = target.copyWith(followersCount: newFollowers);
    });

    try {
      await AuthService.instance.toggleFollow(
        targetUserId: target.id,
        currentlyFollowing: wasFollowing,
      );

      // opcional: refrescar flags desde backend (por si backend calcula isFollowing distinto)
      //await _loadData();
    } catch (e) {
      debugPrint('Error follow/unfollow: $e');

      // Si token expiró => logout y login
      if (_looksLikeUnauthorized(e)) {
        await _forceToLogin();
        return;
      }

      if (!mounted) return;

      // rollback
      final current = _user;
      if (current == null) return;

      setState(() {
        _isFollowing = wasFollowing;
        final rollbackFollowers =
            (current.followersCount + (wasFollowing ? 1 : -1)).clamp(0, 999999);
        _user = current.copyWith(followersCount: rollbackFollowers);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    if (_user == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Perfil')),
        body: RefreshIndicator(
          onRefresh: _loadData,
          child: ListView(
            children: const [
              SizedBox(height: 200),
              Center(child: Text('No se pudo cargar el perfil')),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      body: RefreshIndicator(
        onRefresh: _loadData,
        child: ProfileView(
          posts: _posts,
          currentUser: _user,
          isMe: _isMe,
          isFollowing: _isFollowing,
          onToggleFollow: _isMe ? null : _toggleFollow,
          viewerUsername: null, // viewer real no disponible aquí
          onLogout: null,
          onProfileUpdated: null,
        ),
      ),
    );
  }
}
