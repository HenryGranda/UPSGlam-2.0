// lib/services/filter_service.dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

import 'api_config.dart';

/// Service for applying CUDA filters through the API Gateway
class FilterService {
  FilterService._();
  static final FilterService instance = FilterService._();

  /// Apply filter to an image file
  /// 
  /// Args:
  ///   imageFile: Original image file
  ///   filterName: Filter code (gaussian, prewitt, cr7, etc.)
  /// 
  /// Returns: Path to the filtered image file (saved in temp directory)
  /// 
  /// Throws: Exception if filter application fails
  Future<String> applyFilter({
    required File imageFile,
    required String filterName,
  }) async {
    try {
      final baseUrl = await ApiConfig.requireBaseUrl();
      final uri = Uri.parse('$baseUrl/filters/$filterName');
      
      // Read image bytes
      final imageBytes = await imageFile.readAsBytes();
      
      print('[FilterService] Applying filter: $filterName');
      print('[FilterService] Request URL: $uri');
      print('[FilterService] Image size: ${imageBytes.length} bytes');
      
      // Send POST request with binary image data
      final response = await http.post(
        uri,
        headers: {
          'Content-Type': 'image/jpeg',
        },
        body: imageBytes,
      ).timeout(
        const Duration(seconds: 30),
        onTimeout: () {
          throw Exception('Filter request timeout after 30 seconds');
        },
      );
      
      print('[FilterService] Response status: ${response.statusCode}');
      print('[FilterService] Response headers: ${response.headers}');
      
      if (response.statusCode != 200) {
        throw Exception(
          'Filter application failed: ${response.statusCode} - ${response.body}',
        );
      }
      
      // Get filtered image bytes
      final filteredBytes = response.bodyBytes;
      print('[FilterService] Filtered image size: ${filteredBytes.length} bytes');
      
      // Save filtered image to temporary file
      final tempDir = await getTemporaryDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final extension = path.extension(imageFile.path);
      final filteredFileName = 'filtered_${filterName}_$timestamp$extension';
      final filteredFile = File(path.join(tempDir.path, filteredFileName));
      
      await filteredFile.writeAsBytes(filteredBytes);
      print('[FilterService] Filtered image saved to: ${filteredFile.path}');
      
      return filteredFile.path;
      
    } catch (e) {
      print('[FilterService] Error applying filter: $e');
      rethrow;
    }
  }
  
  /// Get list of available filters (optional - for future use)
  Future<List<Map<String, dynamic>>> getAvailableFilters() async {
    try {
      final baseUrl = await ApiConfig.requireBaseUrl();
      final uri = Uri.parse('$baseUrl/filters');
      
      final response = await http.get(uri);
      
      if (response.statusCode == 200) {
        final data = Map<String, dynamic>.from(
          jsonDecode(response.body),
        );
        final filters = List<Map<String, dynamic>>.from(
          data['filters'] ?? [],
        );
        return filters;
      }
      
      return [];
    } catch (e) {
      print('[FilterService] Error fetching filters: $e');
      return [];
    }
  }
}
