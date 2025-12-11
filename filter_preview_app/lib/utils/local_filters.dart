import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';
import 'dart:math' as math;

/// Filtros locales que implementan algoritmos similares a los filtros CUDA
/// Para preview en tiempo real con precisión mejorada
class LocalFilters {
  
  /// Aplica filtro Prewitt OPTIMIZADO para tiempo real
  /// Versión simplificada con gain reducido
  static img.Image applyPrewitt(img.Image image, {double gain = 3.0}) {
    // Convertir a escala de grises
    final gray = img.grayscale(image);
    final w = gray.width;
    final h = gray.height;
    final result = img.Image(width: w, height: h);
    
    // Procesar directamente sin buffers intermedios (más rápido)
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        // Kernel Prewitt X: [-1 0 1; -1 0 1; -1 0 1]
        final gx = 
          -gray.getPixel(x - 1, y - 1).r.toDouble() +
           gray.getPixel(x + 1, y - 1).r.toDouble() +
          -gray.getPixel(x - 1, y).r.toDouble() +
           gray.getPixel(x + 1, y).r.toDouble() +
          -gray.getPixel(x - 1, y + 1).r.toDouble() +
           gray.getPixel(x + 1, y + 1).r.toDouble();
        
        // Kernel Prewitt Y: [-1 -1 -1; 0 0 0; 1 1 1]
        final gy = 
          -gray.getPixel(x - 1, y - 1).r.toDouble() +
          -gray.getPixel(x, y - 1).r.toDouble() +
          -gray.getPixel(x + 1, y - 1).r.toDouble() +
           gray.getPixel(x - 1, y + 1).r.toDouble() +
           gray.getPixel(x, y + 1).r.toDouble() +
           gray.getPixel(x + 1, y + 1).r.toDouble();
        
        // Magnitud combinada con gain reducido
        final mag = (gx.abs() + gy.abs()) * gain * 0.1;
        final v = mag.clamp(0, 255).toInt();
        result.setPixelRgba(x, y, v, v, v, 255);
      }
    }
    
    return result;
  }
  
  /// Aplica filtro Laplacian OPTIMIZADO para tiempo real
  static img.Image applyLaplacian(img.Image image, {double gain = 2.0}) {
    final gray = img.grayscale(image);
    final w = gray.width;
    final h = gray.height;
    final result = img.Image(width: w, height: h);
    
    // Kernel Laplacian optimizado: [-1 -1 -1; -1 8 -1; -1 -1 -1]
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        final center = gray.getPixel(x, y).r.toDouble() * 8;
        final neighbors = 
          gray.getPixel(x - 1, y - 1).r.toDouble() +
          gray.getPixel(x, y - 1).r.toDouble() +
          gray.getPixel(x + 1, y - 1).r.toDouble() +
          gray.getPixel(x - 1, y).r.toDouble() +
          gray.getPixel(x + 1, y).r.toDouble() +
          gray.getPixel(x - 1, y + 1).r.toDouble() +
          gray.getPixel(x, y + 1).r.toDouble() +
          gray.getPixel(x + 1, y + 1).r.toDouble();
        
        final v = ((center - neighbors) * gain * 0.1).clamp(0, 255).toInt();
        result.setPixelRgba(x, y, v, v, v, 255);
      }
    }
    
    return result;
  }
  
  /// Aplica Gaussian Blur
  static img.Image applyGaussianBlur(img.Image image) {
    return img.gaussianBlur(image, radius: 5);
  }
  
  /// Aplica Box Blur
  static img.Image applyBoxBlur(img.Image image) {
    return img.convolution(
      image,
      filter: [
        1/9, 1/9, 1/9,
        1/9, 1/9, 1/9,
        1/9, 1/9, 1/9,
      ],
    );
  }
  
  /// Aplica efecto UPS Logo (aura dorada)
  static img.Image applyUPSLogo(img.Image image) {
    final result = img.Image.from(image);
    
    for (int y = 0; y < result.height; y++) {
      for (int x = 0; x < result.width; x++) {
        final pixel = result.getPixel(x, y);
        
        // Luminancia
        final lum = (0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b) / 255.0;
        
        // Aplicar tinte dorado UPS (242, 169, 0)
        final factor = 0.25 * lum;
        final r = (pixel.r + (242 - pixel.r) * factor).clamp(0, 255).toInt();
        final g = (pixel.g + (169 - pixel.g) * factor).clamp(0, 255).toInt();
        final b = (pixel.b * (1 - factor * 0.3)).clamp(0, 255).toInt();
        
        result.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    
    return result;
  }
  
  /// Aplica efecto UPS Color (tinte dorado intenso)
  static img.Image applyUPSColor(img.Image image) {
    final result = img.Image.from(image);
    
    for (int y = 0; y < result.height; y++) {
      for (int x = 0; x < result.width; x++) {
        final pixel = result.getPixel(x, y);
        
        final factor = 0.4;
        final r = (pixel.r + (242 - pixel.r) * factor).clamp(0, 255).toInt();
        final g = (pixel.g + (169 - pixel.g) * factor).clamp(0, 255).toInt();
        final b = (pixel.b * 0.7).clamp(0, 255).toInt();
        
        result.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    
    return result;
  }
  
  /// Aplica efecto Boomerang (saturación y brillo)
  static img.Image applyBoomerang(img.Image image) {
    final result = img.Image.from(image);
    
    for (int y = 0; y < result.height; y++) {
      for (int x = 0; x < result.width; x++) {
        final pixel = result.getPixel(x, y);
        
        // Aumentar saturación y brillo
        final r = (pixel.r * 1.2).clamp(0, 255).toInt();
        final g = (pixel.g * 1.05).clamp(0, 255).toInt();
        final b = (pixel.b * 1.3).clamp(0, 255).toInt();
        
        result.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    
    return result;
  }
  
  /// Procesa imagen con el filtro seleccionado
  static Future<Uint8List?> processImageBytes(Uint8List bytes, String filterId) async {
    if (filterId == 'none') return null;
    
    try {
      img.Image? image = img.decodeImage(bytes);
      if (image == null) return null;
      
      // Reducir tamaño más agresivamente para filtros de bordes (más pesados)
      if (filterId == 'prewitt' || filterId == 'laplacian') {
        if (image.width > 320) {
          image = img.copyResize(image, width: 320);
        }
      } else {
        // Otros filtros pueden ser un poco más grandes
        if (image.width > 480) {
          image = img.copyResize(image, width: 480);
        }
      }
      
      img.Image? filtered;
      
      switch (filterId) {
        case 'prewitt':
          filtered = applyPrewitt(image);
          break;
        case 'laplacian':
          filtered = applyLaplacian(image);
          break;
        case 'gaussian':
          filtered = applyGaussianBlur(image);
          break;
        case 'box_blur':
          filtered = applyBoxBlur(image);
          break;
        case 'ups_logo':
          filtered = applyUPSLogo(image);
          break;
        case 'ups_color':
          filtered = applyUPSColor(image);
          break;
        case 'boomerang':
          filtered = applyBoomerang(image);
          break;
        default:
          return null;
      }
      
      if (filtered == null) return null;
      
      return Uint8List.fromList(img.encodeJpg(filtered, quality: 80));
    } catch (e) {
      print('Error processing filter: $e');
      return null;
    }
  }
  
  /// Retorna nombre display del filtro
  static String getFilterDisplayName(String filter) {
    const filterNames = {
      'gaussian': 'Gaussian Blur',
      'box_blur': 'Box Blur',
      'prewitt': 'Prewitt (Bordes)',
      'laplacian': 'Laplacian (Bordes)',
      'ups_logo': 'UPS Logo',
      'ups_color': 'UPS Color',
      'boomerang': 'Boomerang',
    };
    return filterNames[filter] ?? filter;
  }
}
