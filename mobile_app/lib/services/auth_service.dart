import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import 'api_config.dart';

/// Servicio de Autenticación conectado al microservicio Auth (WebFlux + Firebase)
class AuthService {
  AuthService._();
  static final AuthService instance = AuthService._();

  static const _keyIdToken = 'auth_id_token';
  static const _keyRefreshToken = 'auth_refresh_token';
  static const _keyUser = 'auth_user_json';

  /// ==========================
  /// LOGIN: /auth/login
  /// identifier = email o username
  /// ==========================
  Future<void> loginWithEmailPassword(
      String identifier, String password) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/auth/login');

    final res = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'identifier': identifier,
        'password': password,
      }),
    );

    if (res.statusCode != 200) {
      String msg = 'Error ${res.statusCode} al iniciar sesión';
      try {
        final data = jsonDecode(res.body);
        if (data is Map && data['message'] is String) {
          msg = data['message'];
        }
      } catch (_) {}
      throw Exception(msg);
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    final user = data['user'] as Map<String, dynamic>;
    final token = data['token'] as Map<String, dynamic>;

    final idToken = token['idToken'] as String?;
    final refreshToken = token['refreshToken'] as String?;

    if (idToken == null || refreshToken == null) {
      throw Exception('Respuesta de login incompleta (falta token)');
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyIdToken, idToken);
    await prefs.setString(_keyRefreshToken, refreshToken);
    await prefs.setString(_keyUser, jsonEncode(user));
  }

  /// ==========================
  /// REGISTER: /auth/register
  /// Necesita email, password, fullName, username
  /// ==========================
  Future<void> registerWithEmailPassword(
    String email,
    String password,
    String fullName,
    String username,
  ) async {
    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/auth/register');

    final res = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'email': email,
        'password': password,
        'fullName': fullName,
        'username': username,
      }),
    );

    if (res.statusCode >= 200 && res.statusCode < 300) {
      return;
    }
    
    String msg = 'Error ${res.statusCode} al registrarse';
    try {
      final data = jsonDecode(res.body);
      if (data is Map && data['message'] is String) {
        msg = data['message'];
      }
    } catch (_) {}
    throw Exception(msg);
  }

  /// ==========================
  /// GET /auth/me  → perfil actual
  /// ==========================
  Future<Map<String, dynamic>?> fetchCurrentUser() async {
    final idToken = await getIdToken();
    if (idToken == null) return null;

    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/auth/me');

    final res = await http.get(
      uri,
      headers: {
        'Authorization': 'Bearer $idToken',
      },
    );

    if (res.statusCode == 401) {
      // Sesión expirada o inválida
      await logout();
      return null;
    }

    if (res.statusCode != 200) {
      String msg = 'Error ${res.statusCode} al obtener el perfil';
      try {
        final data = jsonDecode(res.body);
        if (data is Map && data['message'] is String) {
          msg = data['message'];
        }
      } catch (_) {}
      throw Exception(msg);
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;

    // Actualizamos copia local
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyUser, jsonEncode(data));

    return data;
  }

  /// ==========================
  /// PATCH /users/me → actualizar perfil (si luego lo usas)
  /// ==========================
  Future<Map<String, dynamic>> updateProfile({
    String? username,
    String? fullName,
    String? bio,
  }) async {
    final idToken = await getIdToken();
    if (idToken == null) {
      throw Exception('No hay sesión activa');
    }

    final payload = <String, dynamic>{};
    if (username != null) payload['username'] = username;
    if (fullName != null) payload['fullName'] = fullName;
    if (bio != null) payload['bio'] = bio;

    final baseUrl = await ApiConfig.requireBaseUrl();
    final uri = Uri.parse('$baseUrl/users/me');

    final res = await http.patch(
      uri,
      headers: {
        'Authorization': 'Bearer $idToken',
        'Content-Type': 'application/json',
      },
      body: jsonEncode(payload),
    );

    if (res.statusCode != 200) {
      String msg = 'Error ${res.statusCode} al actualizar perfil';
      try {
        final data = jsonDecode(res.body);
        if (data is Map && data['message'] is String) {
          msg = data['message'];
        }
      } catch (_) {}
      throw Exception(msg);
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyUser, jsonEncode(data));

    return data;
  }

  /// ==========================
  /// Helpers de sesión local
  /// ==========================

  Future<String?> getIdToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyIdToken);
  }

  Future<Map<String, dynamic>?> getStoredUser() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonStr = prefs.getString(_keyUser);
    if (jsonStr == null) return null;
    return jsonDecode(jsonStr) as Map<String, dynamic>;
  }

  Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_keyIdToken);
    await prefs.remove(_keyRefreshToken);
    await prefs.remove(_keyUser);
  }

  /// ==========================
  /// Stubs para Google / Facebook (opcional)
  /// ==========================
  Future<void> loginWithGoogle() async {
    throw UnimplementedError(
        'Login con Google aún no implementado en AuthService.');
  }

  Future<void> loginWithFacebook() async {
    throw UnimplementedError(
        'Login con Facebook aún no implementado en AuthService.');
  }

  Future<String?> getUserId() async {
    final user = await getStoredUser();
    if (user == null) return null;
    final id = user['id'];
    if (id is String && id.isNotEmpty) return id;
    return null;
  }


}
