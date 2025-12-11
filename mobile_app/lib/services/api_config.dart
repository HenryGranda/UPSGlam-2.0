import 'package:shared_preferences/shared_preferences.dart';

class ApiConfig {
  static const String _keyBaseUrl = 'backend_base_url';

  /// Guarda la URL base del backend.
  /// Ejemplo recomendado: http://192.168.1.10:8080/api
  static Future<void> setBaseUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyBaseUrl, url.trim());
  }

  /// Devuelve la URL base cruda (tal cual se guardó) o null si no hay.
  static Future<String?> getBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyBaseUrl);
  }

  /// Devuelve la URL base normalizada o lanza excepción si no está configurada.
  ///
  /// - Asegura que NO termine en slash extra.
  /// - Tú deberías guardar algo como: http://192.168.1.10:8080/api
  static Future<String> requireBaseUrl() async {
    final raw = await getBaseUrl();
    if (raw == null || raw.trim().isEmpty) {
      throw Exception(
        'Backend no configurado. Ve a "Configurar IP del Backend" primero.',
      );
    }

    // quitamos slashes extra al final
    final normalized = raw.trim().replaceAll(RegExp(r'/+$'), '');
    return normalized;
  }
}
