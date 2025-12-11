import 'dart:typed_data';
import 'package:http/http.dart' as http;

class FilterService {
  // ⚠️ IMPORTANTE: Cambia esta URL a la IP de tu servidor
  // Ejemplos:
  // - Localhost (emulador Android): 'http://10.0.2.2:5000'
  // - Localhost (dispositivo real): 'http://TU_IP_LOCAL:5000'
  // - Servidor remoto: 'http://TU_SERVER:5000'
  static const String baseUrl = 'http://localhost:5000';

  /// Aplica filtro enviando imagen al backend cuda-lab-back
  /// 
  /// [imageBytes] - Imagen original sin filtro
  /// [filterName] - Nombre del filtro a aplicar
  /// 
  /// Retorna imagen procesada con PyCUDA o null si hay error
  static Future<Uint8List?> applyFilter(
    Uint8List imageBytes,
    String filterName,
  ) async {
    try {
      final url = Uri.parse('$baseUrl/filters/$filterName');
      
      print('🚀 Enviando imagen al backend: $url');
      print('📦 Tamaño de imagen: ${imageBytes.length} bytes');
      print('🎨 Filtro: $filterName');

      final response = await http.post(
        url,
        headers: {
          'Content-Type': 'image/jpeg',
        },
        body: imageBytes,
      ).timeout(
        const Duration(seconds: 30),
        onTimeout: () {
          throw Exception('Timeout: El servidor tardó demasiado');
        },
      );

      if (response.statusCode == 200) {
        print('✅ Filtro aplicado exitosamente');
        print('📥 Tamaño de respuesta: ${response.bodyBytes.length} bytes');
        return response.bodyBytes;
      } else {
        print('❌ Error del servidor: ${response.statusCode}');
        print('📄 Respuesta: ${response.body}');
        return null;
      }
    } catch (e) {
      print('❌ Error al aplicar filtro: $e');
      return null;
    }
  }

  /// Lista todos los filtros disponibles del backend
  static Future<List<String>> getAvailableFilters() async {
    try {
      final url = Uri.parse('$baseUrl/filters');
      final response = await http.get(url);

      if (response.statusCode == 200) {
        // Parse JSON response
        // TODO: Implementar parseo de respuesta JSON
        return [
          'gaussian',
          'box_blur',
          'prewitt',
          'laplacian',
          'ups_logo',
          'ups_color',
          'boomerang',
        ];
      }
    } catch (e) {
      print('Error al obtener filtros: $e');
    }

    return [];
  }
}
