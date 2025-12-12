import 'comment_model.dart';
import 'avatars.dart';

class PostModel {
  String id;
  String username;
  String avatar;        // aquí guardamos LO QUE VIENE del backend
  String timestamp;
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

  factory PostModel.fromJson(Map<String, dynamic> json) {
    final username = (json['username'] as String?) ?? 'usuario_ups';
    final userPhotoUrl = json['userPhotoUrl'] as String?;

    final createdAt = json['createdAt'];
    String timestamp = 'Ahora';
    if (createdAt is String && createdAt.isNotEmpty) {
      timestamp = createdAt;
    }

    final avatar = userPhotoUrl ?? '';

    return PostModel(
      id: (json['id'] ?? '').toString(),
      username: '@$username',
      avatar: avatar,
      timestamp: timestamp,
      imageUrl: (json['imageUrl'] ?? '') as String,
      caption: (json['description'] ?? '') as String,
      filter: json['filter'] as String?,
      likes: (json['likesCount'] ?? 0) as int,
      liked: (json['likedByMe'] ?? false) as bool,
      comments: [],
    );
  }

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
