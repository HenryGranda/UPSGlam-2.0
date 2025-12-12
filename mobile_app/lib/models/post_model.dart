import 'comment_model.dart';
import 'avatars.dart';
import 'package:intl/intl.dart';

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
  int commentsCount;
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
    this.commentsCount = 0,
    List<CommentModel>? comments,
  }) : comments = comments ?? [];

  factory PostModel.fromJson(Map<String, dynamic> json) {
    final username = (json['username'] as String?) ?? 'usuario_ups';
    final userPhotoUrl = json['userPhotoUrl'] as String?;

    final createdAt = json['createdAt'];
    final timestamp = _formatTimestamp(createdAt);

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
      commentsCount: (json['commentsCount'] ?? 0) as int,
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
      'commentsCount': commentsCount,
    };
  }
}

String _formatTimestamp(dynamic raw) {
  DateTime? dt;

  if (raw is String && raw.isNotEmpty) {
    return raw;
  }

  if (raw is Map) {
    final seconds = raw['seconds'] ?? raw['_seconds'];
    final nanos = raw['nanos'] ?? raw['_nanoseconds'];
    if (seconds is num) {
      dt = DateTime.fromMillisecondsSinceEpoch(
        (seconds.toDouble() * 1000).toInt(),
        isUtc: true,
      );
      if (nanos is num) {
        dt = dt.add(Duration(microseconds: (nanos.toDouble() / 1000).round()));
      }
    }
  } else if (raw is num) {
    dt = DateTime.fromMillisecondsSinceEpoch(raw.toInt(), isUtc: true);
  } else if (raw is List && raw.length >= 3 && raw[0] is int) {
    // formato [yyyy,mm,dd,HH,mm,ss]
    final year = raw[0] as int;
    final month = raw[1] as int;
    final day = raw[2] as int;
    final hour = raw.length > 3 ? raw[3] as int : 0;
    final minute = raw.length > 4 ? raw[4] as int : 0;
    final second = raw.length > 5 ? raw[5] as int : 0;
    dt = DateTime(year, month, day, hour, minute, second);
  }

  if (dt == null) return 'Ahora';

  return DateFormat('dd/MM/yyyy HH:mm').format(dt.toLocal());
}
