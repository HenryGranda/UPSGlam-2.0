import 'comment_model.dart';

class PostModel {
  String id;
  String username;
  String avatar;
  String timestamp;      // texto “Ahora”, “hace 2 h”, etc.
  String imageUrl;
  String caption;
  String? filter;
  int likes;
  bool liked;
  List<CommentModel> comments;

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

  /// --------- NUEVO: construir desde JSON del backend ---------
  ///
  /// Espera algo como el PostResponse de tu post-service:
  /// {
  ///   "id": "...",
  ///   "username": "...",
  ///   "userPhotoUrl": "...",
  ///   "imageUrl": "...",
  ///   "filter": "box_blur",
  ///   "description": "texto",
  ///   "createdAt": "2025-12-10T20:20:20",
  ///   "likesCount": 0,
  ///   "likedByMe": false
  /// }
  factory PostModel.fromJson(Map<String, dynamic> json) {
    final username = (json['username'] as String?) ?? 'usuario_ups';
    final userPhotoUrl = json['userPhotoUrl'] as String?;

    // si no viene fecha, mostramos “Ahora”
    final createdAt = json['createdAt'];
    String timestamp = 'Ahora';
    if (createdAt is String && createdAt.isNotEmpty) {
      // de momento lo mostramos tal cual; si quieres luego formateamos
      timestamp = createdAt;
    }

    return PostModel(
      id: (json['id'] ?? '').toString(),
      username: '@$username',
      avatar:
          userPhotoUrl != null && userPhotoUrl.isNotEmpty
              ? userPhotoUrl
              : 'assets/images/user_profile.png',
      timestamp: timestamp,
      imageUrl: (json['imageUrl'] ?? '') as String,
      caption: (json['description'] ?? '') as String,
      filter: json['filter'] as String?,
      likes: (json['likesCount'] ?? 0) as int,
      liked: (json['likedByMe'] ?? false) as bool,
      comments: [], 
    );
  }

  /// (opcional) por si en algún momento quieres mandar el post al backend
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'username': username,
      'imageUrl': imageUrl,
      'description': caption,
      'filter': filter,
      'likesCount': likes,
      'likedByMe': liked,
    };
  }
}
