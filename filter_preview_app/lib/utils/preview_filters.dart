import 'package:flutter/material.dart';

/// ColorFilters optimizados para preview en tiempo real
/// Estos son aproximaciones visuales rápidas (GPU)
/// La imagen real se procesa con CUDA en el backend
class PreviewFilters {
  
  /// Retorna ColorFilter para preview en tiempo real
  static ColorFilter? getPreviewColorFilter(String filterId) {
    switch (filterId) {
      case 'prewitt':
      case 'laplacian':
        // Detección de bordes: escala de grises con alto contraste
        // Convierte a gris y aumenta contraste para simular detección de bordes
        return const ColorFilter.matrix([
          0.5, 0.5, 0.5, 0, -50,  // R: gris con contraste
          0.5, 0.5, 0.5, 0, -50,  // G: gris con contraste
          0.5, 0.5, 0.5, 0, -50,  // B: gris con contraste
          0,   0,   0,   1, 0,     // A: sin cambio
        ]);
      
      case 'gaussian':
      case 'box_blur':
        // Blur: desaturar ligeramente como indicador visual
        return const ColorFilter.matrix([
          0.6, 0.2, 0.2, 0, 0,
          0.2, 0.6, 0.2, 0, 0,
          0.2, 0.2, 0.6, 0, 0,
          0,   0,   0,   1, 0,
        ]);
      
      case 'ups_logo':
        // Tinte dorado UPS sutil
        return ColorFilter.mode(
          const Color(0xFFF2A900).withOpacity(0.25),
          BlendMode.screen,
        );
      
      case 'ups_color':
        // Tinte dorado más intenso
        return ColorFilter.mode(
          const Color(0xFFF2A900).withOpacity(0.4),
          BlendMode.overlay,
        );
      
      case 'boomerang':
        // Aumentar saturación
        return const ColorFilter.matrix([
          1.3, 0,   0,   0, 0,
          0,   1.1, 0,   0, 0,
          0,   0,   1.4, 0, 0,
          0,   0,   0,   1, 0,
        ]);
      
      default:
        return null;
    }
  }
}
