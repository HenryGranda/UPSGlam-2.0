import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';
import 'api_config.dart';
import 'auth_service.dart';

class AppNotification {
  final String id;
  final String type; // like | comment | follow
  final String? actorUsername;
  final String? actorPhotoUrl;
  final String? postId;
  final String? commentId;
  final DateTime? createdAt;
  final bool read;

  AppNotification({
    required this.id,
    required this.type,
    required this.actorUsername,
    required this.actorPhotoUrl,
    required this.postId,
    required this.commentId,
    required this.createdAt,
    required this.read,
  });

  factory AppNotification.fromJson(Map<String, dynamic> json) {
    return AppNotification(
      id: json['id'] as String? ?? '',
      type: json['type'] as String? ?? '',
      actorUsername: json['actorUsername'] as String?,
      actorPhotoUrl: json['actorPhotoUrl'] as String?,
      postId: json['postId'] as String?,
      commentId: json['commentId'] as String?,
      createdAt: json['createdAt'] != null
          ? DateTime.fromMillisecondsSinceEpoch(
              (json['createdAt'] as num).toInt())
          : null,
      read: json['read'] as bool? ?? false,
    );
  }

  String humanTime() {
    if (createdAt == null) return '';
    final now = DateTime.now();
    final diff = now.difference(createdAt!);
    if (diff.inMinutes < 1) return 'Justo ahora';
    if (diff.inMinutes < 60) return '${diff.inMinutes} min';
    if (diff.inHours < 24) return '${diff.inHours} h';
    return DateFormat('dd/MM/yyyy HH:mm').format(createdAt!);
  }
}

class NotificationService {
  NotificationService._();
  static final NotificationService instance = NotificationService._();

  Future<List<AppNotification>> fetchNotifications({int limit = 30}) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final token = await AuthService.instance.getIdToken();
    final user = await AuthService.instance.getStoredUser();
    final userId = (user?['id'] ?? user?['uid'])?.toString();
    if (token == null || userId == null) {
      throw Exception('Sesión expirada');
    }

    final uri = Uri.parse('$baseUrl/notifications?limit=$limit');
    final res = await http.get(uri, headers: {
      'Authorization': 'Bearer $token',
      'X-User-Id': userId,
    });

    if (res.statusCode != 200) {
      throw Exception('Error ${res.statusCode} al cargar notificaciones');
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    final items = (data['items'] as List<dynamic>? ?? [])
        .map((e) => AppNotification.fromJson(e as Map<String, dynamic>))
        .toList();
    return items;
  }

  Future<void> markAsRead(String id) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final token = await AuthService.instance.getIdToken();
    final user = await AuthService.instance.getStoredUser();
    final userId = (user?['id'] ?? user?['uid'])?.toString();
    if (token == null || userId == null) return;

    final uri = Uri.parse('$baseUrl/notifications/$id/read');
    await http.patch(uri, headers: {
      'Authorization': 'Bearer $token',
      'X-User-Id': userId,
    });
  }
}
