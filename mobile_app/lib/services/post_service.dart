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

  /// =============
  /// HEADERS CON TOKEN
  /// =============
  Future<Map<String, String>> _authHeaders({bool json = false}) async {
    final token = await AuthService.instance.getIdToken();

    // Leemos los datos del usuario desde el storage
    final userData = await AuthService.instance.getStoredUser();
    final userId = (userData?['id'] ?? userData?['uid'])?.toString();
    final username = userData?['username'] as String?;

    final headers = <String, String>{};

    if (json) {
      headers['Content-Type'] = 'application/json';
    }
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
  /// Body: multipart/form-data con campo "image"
  Future<String> uploadImage(File imageFile) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/images/upload'); // gateway: /api/images/upload

    const fieldName = 'image';

    final request = http.MultipartRequest('POST', uri)
      ..files.add(
        await http.MultipartFile.fromPath(
          fieldName,
          imageFile.path,
          contentType: MediaType('image', 'jpeg'),
        ),
      );

    final response = await request.send();
    final body = await response.stream.bytesToString();

    // debug opcional
    print('/images/upload BODY => $body');

    if (response.statusCode != 200 && response.statusCode != 201) {
      throw Exception(
        'Error ${response.statusCode} al subir imagen: $body',
      );
    }

    final json = jsonDecode(body) as Map<String, dynamic>;
    final mediaUrl = json['imageUrl'] as String?;

    if (mediaUrl == null || mediaUrl.isEmpty) {
      throw Exception('Respuesta sin imageUrl válida al subir imagen: $json');
    }

    return mediaUrl;
  }

  /// CREAR POST  (POST /posts)
  Future<Map<String, dynamic>> createPost({
    required File imageFile,
    required String caption,
    String? filter,
    String? username,
    String? userPhotoUrl,
  }) async {
    final baseUrl = await ApiConfig.requireBaseUrl();

    // Asegurarnos de que haya sesión
    final idToken = await AuthService.instance.getIdToken();
    if (idToken == null || idToken.isEmpty) {
      throw Exception('No hay sesión activa para crear post');
    }

    // 1) Subimos imagen
    final mediaUrl = await uploadImage(imageFile);

    // 2) Creamos el post
    final uri = Uri.parse('$baseUrl/posts');

    final body = jsonEncode({
      'caption': caption,
      'filter': filter,
      'mediaUrl': mediaUrl,
      'mediaType': 'image/jpeg',
      'username': username,
      'userPhotoUrl': userPhotoUrl,   
    });

    final resp = await http.post(
      uri,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $idToken',
      },
      body: body,
    );

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

    final headers = await _authHeaders(); // intenta mandar token si hay

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

  /// ==========================
  /// LIKES
  /// ==========================
  Future<void> toggleLike({
    required String postId,
    required bool currentlyLiked,
  }) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/posts/$postId/likes');

    // headers con JSON + Authorization + X-User-Id / X-Username
    final headers = await _authHeaders(json: true);

    late http.Response resp;

    if (currentlyLiked) {
      // ya estaba likeado → UNLIKE
      resp = await http.delete(uri, headers: headers);
    } else {
      // no estaba likeado → LIKE
      resp = await http.post(uri, headers: headers);
    }

    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw Exception(
        'Error ${resp.statusCode} al hacer like: ${resp.body}',
      );
    }
  }

  /// ==========================
  /// COMENTARIOS
  /// ==========================

  Future<List<CommentModel>> fetchComments(String postId) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/posts/$postId/comments');

    // Mandamos token y headers extra
    final headers = await _authHeaders();

    final res = await http.get(uri, headers: headers.isEmpty ? null : headers);

    if (res.statusCode != 200) {
      throw Exception(
        'Error ${res.statusCode} al obtener comentarios: ${res.body}',
      );
    }

    final body = res.body;
    print('🗨️ fetchComments($postId) BODY => $body');

    final decoded = jsonDecode(body);

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
      body: jsonEncode({
        'text': newText,
      }),
    );

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'Error ${res.statusCode} al editar comentario: ${res.body}',
      );
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return CommentModel.fromJson(data);
  }


  /// Devuelve el CommentModel creado por el backend
  Future<CommentModel> addComment({
    required String postId,
    required String text,
  }) async {
    final baseUrl = await ApiConfig.requireBaseUrl();

    final idToken = await AuthService.instance.getIdToken();
    if (idToken == null || idToken.isEmpty) {
      throw Exception('No hay sesión activa para comentar');
    }

    // datos usuario
    final userData = await AuthService.instance.getStoredUser();
    final userId = (userData?['id'] ?? userData?['uid']) as String?;
    final username = (userData?['username'] as String?) ?? 'unknown';
    final userPhotoUrl = userData?['photoUrl'] as String?;

    final uri = Uri.parse('$baseUrl/posts/$postId/comments');

    final headers = <String, String>{
      'Authorization': 'Bearer $idToken',
      'Content-Type': 'application/json',
    };
    if (userId != null && userId.isNotEmpty) {
      headers['X-User-Id'] = userId;
    }

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
  

  /// DELETE /posts/{postId}/comments/{commentId}
  Future<void> deleteComment({
    required String postId,
    required String commentId,
  }) async {
    final baseUrl = await ApiConfig.requireBaseUrl();

    final idToken = await AuthService.instance.getIdToken();
    if (idToken == null || idToken.isEmpty) {
      throw Exception('No hay sesión activa para eliminar comentario');
    }

    final userData = await AuthService.instance.getStoredUser();
    final userId = (userData?['id'] ?? userData?['uid'])?.toString();
    final username = userData?['username'] as String?;

    final uri = Uri.parse('$baseUrl/posts/$postId/comments/$commentId');

    final headers = <String, String>{
      'Authorization': 'Bearer $idToken',
      'Content-Type': 'application/json',
    };

    if (userId != null && userId.isNotEmpty) {
      headers['X-User-Id'] = userId;      // debe coincidir con el que se guardó
    }
    if (username != null && username.isNotEmpty) {
      headers['X-Username'] = username;
    }

    final res = await http.delete(uri, headers: headers);

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'Error ${res.statusCode} al eliminar comentario: ${res.body}',
      );
    }
  }
}
