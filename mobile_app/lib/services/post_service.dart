import 'dart:io';
import 'package:http/http.dart' as http;
import 'api_config.dart';

class PostService {
  static Future<String> sendToCuda(File imageFile, String filterName) async {
    // 1. Obtener la URL base que guardas desde IpConfigScreen
    final baseUrl = await ApiConfig.requireBaseUrl();

    // OJO: aquí asumo que tu microservicio CUDA expone /process-image
    final uri = Uri.parse('$baseUrl/process-image');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('image', imageFile.path))
      ..fields['filter'] = filterName;

    final response = await request.send();

    if (response.statusCode != 200) {
      final body = await response.stream.bytesToString();
      throw Exception(
          'Error ${response.statusCode} al procesar imagen: $body');
    }

    final resString = await response.stream.bytesToString();
    // Aquí asumo que el backend devuelve directamente la URL como texto.
    // Si devuelve JSON, habría que parsear.
    return resString;
  }
}
