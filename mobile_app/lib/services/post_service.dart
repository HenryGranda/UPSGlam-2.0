import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import 'package:mobile_app/models/comment_model.dart';
import 'api_config.dart';
import 'auth_service.dart';

class PostService {
  PostService._();
  static final PostService instance = PostService._();

  /// HEADERS CON TOKEN + X-User-Id/Username
  Future<Map<String, String>> _authHeaders({bool json = false}) async {
    final token = await AuthService.instance.getIdToken();

    final userData = await AuthService.instance.getStoredUser();
    final userId = (userData?['id'] ?? userData?['uid'])?.toString();
    final username = userData?['username'] as String?;

    final headers = <String, String>{};
    if (json) headers['Content-Type'] = 'application/json';
    if (token != null && token.isNotEmpty) {
      headers['Authorization'] = 'Bearer $token';
    }
    if (userId != null && userId.isNotEmpty) {
      headers['X-User-Id'] = userId;
    }
    if (username != null && username.isNotEmpty) {
      headers['X-Username'] = username;
    }
    return headers;
  }

  /// SUBIR IMAGEN A SUPABASE (POST /images/upload)
  Future<String> uploadImage(File imageFile) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/images/upload');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(
        await http.MultipartFile.fromPath(
          'image',
          imageFile.path,
          contentType: MediaType('image', 'jpeg'),
        ),
      );

    final response = await request.send();
    final body = await response.stream.bytesToString();

    if (response.statusCode != 200 && response.statusCode != 201) {
      throw Exception('Error ${response.statusCode} al subir imagen: $body');
    }

    final json = jsonDecode(body) as Map<String, dynamic>;
    final mediaUrl = json['imageUrl'] as String?;
    if (mediaUrl == null || mediaUrl.isEmpty) {
      throw Exception('Respuesta sin imageUrl valida al subir imagen: $json');
    }
    return mediaUrl;
  }

  /// CREAR POST (POST /posts)
  Future<Map<String, dynamic>> createPost({
    required File imageFile,
    required String caption,
    String? filter,
    String? username,
    String? userPhotoUrl,
  }) async {
    final baseUrl = await ApiConfig.requireBaseUrl();

    final idToken = await AuthService.instance.getIdToken();
    if (idToken == null || idToken.isEmpty) {
      throw Exception('No hay sesion activa para crear post');
    }

    final headers = await _authHeaders(json: true);

    final mediaUrl = await uploadImage(imageFile);

    final uri = Uri.parse('$baseUrl/posts');

    final body = jsonEncode({
      'caption': caption,
      'filter': filter,
      'mediaUrl': mediaUrl,
      'mediaType': 'image/jpeg',
      'username': username,
      'userPhotoUrl': userPhotoUrl,
    });

    final resp = await http.post(uri, headers: headers, body: body);

    if (resp.statusCode != 200 && resp.statusCode != 201) {
      throw Exception(
        'Error ${resp.statusCode} al crear post: ${resp.body}',
      );
    }

    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  /// FEED (GET /feed)
  Future<List<Map<String, dynamic>>> fetchFeed() async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/feed?page=0&size=20');

    final headers = await _authHeaders();
    final resp = await http.get(uri, headers: headers.isEmpty ? null : headers);

    if (resp.statusCode != 200) {
      throw Exception(
        'Error ${resp.statusCode} al obtener feed: ${resp.body}',
      );
    }

    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    final posts =
        (data['posts'] as List<dynamic>? ?? []).cast<Map<String, dynamic>>();
    return posts;
  }

  /// LIKES
  Future<void> toggleLike({
    required String postId,
    required bool currentlyLiked,
  }) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/posts/$postId/likes');

    final headers = await _authHeaders(json: true);
    late http.Response resp;

    if (currentlyLiked) {
      resp = await http.delete(uri, headers: headers);
    } else {
      resp = await http.post(uri, headers: headers);
    }

    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw Exception(
        'Error ${resp.statusCode} al hacer like: ${resp.body}',
      );
    }
  }

  /// COMENTARIOS
  Future<List<CommentModel>> fetchComments(String postId) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/posts/$postId/comments');

    final headers = await _authHeaders();
    final res = await http.get(uri, headers: headers.isEmpty ? null : headers);

    if (res.statusCode != 200) {
      throw Exception(
        'Error ${res.statusCode} al obtener comentarios: ${res.body}',
      );
    }

    final decoded = jsonDecode(res.body);
    List<dynamic> rawList;
    if (decoded is List) {
      rawList = decoded;
    } else if (decoded is Map<String, dynamic>) {
      rawList = (decoded['comments'] as List?) ?? <dynamic>[];
    } else {
      throw Exception('Formato inesperado de respuesta de comentarios: $decoded');
    }

    return rawList
        .map((e) => CommentModel.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<CommentModel> updateComment({
    required String postId,
    required String commentId,
    required String newText,
  }) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final headers = await _authHeaders(json: true);

    final uri = Uri.parse('$baseUrl/posts/$postId/comments/$commentId');

    final res = await http.put(
      uri,
      headers: headers,
      body: jsonEncode({'text': newText}),
    );

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'Error ${res.statusCode} al editar comentario: ${res.body}',
      );
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return CommentModel.fromJson(data);
  }

  Future<CommentModel> addComment({
    required String postId,
    required String text,
  }) async {
    final baseUrl = await ApiConfig.requireBaseUrl();

    final idToken = await AuthService.instance.getIdToken();
    if (idToken == null || idToken.isEmpty) {
      throw Exception('No hay sesion activa para comentar');
    }

    final userData = await AuthService.instance.getStoredUser();
    final userId = (userData?['id'] ?? userData?['uid']) as String?;
    final username = (userData?['username'] as String?) ?? 'unknown';
    final userPhotoUrl = userData?['photoUrl'] as String?;

    final uri = Uri.parse('$baseUrl/posts/$postId/comments');

    final headers = await _authHeaders(json: true);

    final res = await http.post(
      uri,
      headers: headers,
      body: jsonEncode({
        'text': text,
        'username': username,
        'userPhotoUrl': userPhotoUrl,
      }),
    );

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'Error ${res.statusCode} al crear comentario: ${res.body}',
      );
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return CommentModel.fromJson(data);
  }

  Future<void> deleteComment({
    required String postId,
    required String commentId,
  }) async {
    final baseUrl = await ApiConfig.requireBaseUrl();

    final headers = await _authHeaders(json: true);

    final uri = Uri.parse('$baseUrl/posts/$postId/comments/$commentId');

    final res = await http.delete(uri, headers: headers);

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'Error ${res.statusCode} al eliminar comentario: ${res.body}',
      );
    }
  }

  /// PATCH /posts/{postId}/caption
  Future<void> updatePostCaption({
    required String postId,
    required String newCaption,
  }) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final headers = await _authHeaders(json: true);

    final uri = Uri.parse('$baseUrl/posts/$postId/caption');

    final res = await http.patch(
      uri,
      headers: headers,
      body: jsonEncode({'caption': newCaption}),
    );

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'Error ${res.statusCode} al actualizar caption: ${res.body}',
      );
    }
  }

  /// DELETE /posts/{postId}
  Future<void> deletePost(String postId) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final headers = await _authHeaders(json: true);

    final uri = Uri.parse('$baseUrl/posts/$postId');

    final res = await http.delete(uri, headers: headers);

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'Error ${res.statusCode} al eliminar post: ${res.body}',
      );
    }
  }

  Future<List<dynamic>> fetchPostsByUsername(String username) async {
    final baseUrl = await ApiConfig.requireBaseUrl();

    final idToken = await AuthService.instance.getIdToken();
    if (idToken == null || idToken.isEmpty) {
      throw Exception('No hay sesion activa');
    }

    final clean = username.startsWith('@') ? username.substring(1) : username;

    final uri = Uri.parse('$baseUrl/posts/user/$clean');

    final res = await http.get(
      uri,
      headers: {
        'Authorization': 'Bearer $idToken',
        'Content-Type': 'application/json',
      },
    );

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'Error ${res.statusCode} al cargar posts de usuario: ${res.body}',
      );
    }

    final decoded = jsonDecode(res.body);

    if (decoded is Map<String, dynamic> && decoded['posts'] is List) {
      return decoded['posts'] as List<dynamic>;
    }

    throw Exception('Respuesta inesperada: ${res.body}');
  }
}
