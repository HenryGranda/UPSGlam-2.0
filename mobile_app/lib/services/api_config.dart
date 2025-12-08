import 'package:shared_preferences/shared_preferences.dart';

class ApiConfig {
  static const String _keyBaseUrl = 'backend_base_url';

  /// Guarda la URL base, por ejemplo:
  /// http://192.168.0.10:8082/api
  static Future<void> setBaseUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyBaseUrl, url.trim());
  }

  /// Obtiene la URL base (puede ser null si no se ha configurado)
  static Future<String?> getBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyBaseUrl);
  }

  /// Versión que OBLIGA a que exista URL. Si no está configurada, lanza excepción.
  static Future<String> requireBaseUrl() async {
    final url = await getBaseUrl();
    if (url == null || url.isEmpty) {
      throw Exception('Backend no configurado. Ve a pantalla de IP.');
    }
    return url;
  }
}
