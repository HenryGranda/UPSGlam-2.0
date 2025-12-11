import 'package:intl/intl.dart';

class CommentModel {
  final String id;
  final String username;
  final String avatar;
  final String text;
  final String timestamp; // texto ya formateado

  CommentModel({
    required this.id,
    required this.username,
    required this.avatar,
    required this.text,
    required this.timestamp,
  });

  factory CommentModel.fromJson(Map<String, dynamic> json) {
    final createdAt = _parseCreatedAt(json['createdAt']);
    return CommentModel(
      id: (json['id'] ?? json['iid'] ?? '') as String,
      username: json['username'] ?? 'unknown',
      avatar: json['userPhotoUrl'] ?? 'assets/images/user_profile.png',
      text: json['text'] ?? '',
      timestamp: _formatDate(createdAt),
    );
  }
}

/// Parsea el array [yyyy, mm, dd, HH, mm, ss, ...] -> DateTime
DateTime? _parseCreatedAt(dynamic raw) {
  if (raw is List && raw.length >= 3) {
    final year = raw[0] as int;
    final month = raw[1] as int;
    final day = raw[2] as int;
    final hour = raw.length > 3 ? raw[3] as int : 0;
    final minute = raw.length > 4 ? raw[4] as int : 0;
    final second = raw.length > 5 ? raw[5] as int : 0;

    // lo tomamos tal cual como hora local
    return DateTime(year, month, day, hour, minute, second);
  }
  return null;
}

/// Devuelve algo como "11/12/2025 20:31"
String _formatDate(DateTime? dt) {
  if (dt == null) return 'Ahora';
  final formatter = DateFormat('dd/MM/yyyy HH:mm');
  return formatter.format(dt);
}
