import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'dart:typed_data';
import 'dart:math' as math;

/// Filtros locales que simulan los filtros oficiales del backend CUDA.
/// Útiles únicamente para previsualizar efectos en tiempo real.
class LocalFilters {
  static bool _assetsLoaded = false;
  static img.Image? _sonrisaTexture;
  static final Map<int, img.Image> _resizedTextureCache = {};
  static final List<_BoomerangBall> _boomerangBalls = [];
  static double _boomerangWidth = 0;
  static double _boomerangHeight = 0;
  static int _lastBoomerangUpdateMs = DateTime.now().millisecondsSinceEpoch;
  static final math.Random _random = math.Random();
  static int _numBoomerangBalls = 0;

  static Future<void> ensureInitialized() async {
    if (_assetsLoaded) return;
    try {
      final data = await rootBundle.load('assets/sonrisa.png');
      final bytes = data.buffer.asUint8List();
      _sonrisaTexture = img.decodeImage(bytes);
    } catch (_) {
      _sonrisaTexture = null;
    } finally {
      _assetsLoaded = true;
    }
  }
  
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
  static img.Image applyGaussian(img.Image image) {
    return img.gaussianBlur(image, radius: 5);
  }
  
  /// Aplica efecto Blox Blur (pixelado por bloques)
  static img.Image applyBloxBlur(img.Image image, {int blockSize = 12}) {
    final result = img.Image.from(image);
    final h = result.height;
    final w = result.width;

    for (int y = 0; y < h; y += blockSize) {
      final yEnd = math.min(y + blockSize, h);
      for (int x = 0; x < w; x += blockSize) {
        final xEnd = math.min(x + blockSize, w);

        double rSum = 0, gSum = 0, bSum = 0;
        final count = (yEnd - y) * (xEnd - x);

        for (int yy = y; yy < yEnd; yy++) {
          for (int xx = x; xx < xEnd; xx++) {
            final pixel = result.getPixel(xx, yy);
            rSum += pixel.r.toDouble();
            gSum += pixel.g.toDouble();
            bSum += pixel.b.toDouble();
          }
        }

        final avgR = (rSum / count).round().clamp(0, 255).toInt();
        final avgG = (gSum / count).round().clamp(0, 255).toInt();
        final avgB = (bSum / count).round().clamp(0, 255).toInt();

        for (int yy = y; yy < yEnd; yy++) {
          for (int xx = x; xx < xEnd; xx++) {
            result.setPixelRgba(xx, yy, avgR, avgG, avgB, 255);
          }
        }
      }
    }

    return result;
  }

  /// Versión clásica de box blur (se mantiene como respaldo)
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
  
  /// Aplica efecto Don Bosco (aura dorada + tinte suave)
  static img.Image applyUpsLogo(img.Image image) {
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
  
  /// Aplica efecto Boomerang simulando las pelotas luminosas del filtro CUDA
  static img.Image applyBoomerang(img.Image image, {int numOrbs = 3}) {
    final result = img.Image.from(image);
    final minSide = math.min(result.width, result.height).toDouble();
    final radius = math.max(14, math.min(90, (minSide * 0.07))).toDouble();

    _updateBoomerangState(result.width.toDouble(), result.height.toDouble(), numOrbs, radius);

    // Aumentar saturación base
    for (final pixel in result) {
      pixel
        ..r = (pixel.r * 1.05 + 10).clamp(0, 255).toInt()
        ..g = (pixel.g * 1.03 + 5).clamp(0, 255).toInt()
        ..b = (pixel.b * 1.08 + 8).clamp(0, 255).toInt();
    }

    final textureBase = _sonrisaTexture;
    img.Image? cachedTex;

    for (final ball in _boomerangBalls) {
      img.Image? overlay;
      if (textureBase != null) {
        overlay = _getResizedTexture(textureBase!, radius);
      }

      final dstX = (ball.x - radius).round();
      final dstY = (ball.y - radius).round();

      if (overlay != null) {
        img.compositeImage(
          result,
          overlay!,
          dstX: dstX,
          dstY: dstY,
          blend: img.BlendMode.alpha,
        );
      } else {
        _paintFallbackOrb(result, ball.x, ball.y, radius);
      }

      // trail tenue
      if (overlay != null) {
        final trailX = (ball.x - ball.vx * 0.02 - radius * 0.7).round();
        final trailY = (ball.y - ball.vy * 0.02 - radius * 0.7).round();
        final smaller = textureBase != null
            ? _getResizedTexture(textureBase!, radius * 0.75)
            : null;
        if (smaller != null) {
          final trailTexture = smaller;
          img.compositeImage(
            result,
            trailTexture,
            dstX: trailX,
            dstY: trailY,
            blend: img.BlendMode.alpha,
          );
        }
      }
    }

    return result;
  }

  static img.Image? _getResizedTexture(img.Image base, double radius) {
    final size = (radius * 2).round().clamp(4, 512);
    if (_resizedTextureCache.containsKey(size)) {
      return _resizedTextureCache[size];
    }
    final resized = img.copyResize(base, width: size, height: size, interpolation: img.Interpolation.cubic);
    if (_resizedTextureCache.length > 6) {
      _resizedTextureCache.remove(_resizedTextureCache.keys.first);
    }
    _resizedTextureCache[size] = resized;
    return resized;
  }

  static void _paintFallbackOrb(img.Image image, double cx, double cy, double radius) {
    final minX = math.max(0, (cx - radius).floor());
    final maxX = math.min(image.width - 1, (cx + radius).ceil());
    final minY = math.max(0, (cy - radius).floor());
    final maxY = math.min(image.height - 1, (cy + radius).ceil());

    for (int y = minY; y <= maxY; y++) {
      for (int x = minX; x <= maxX; x++) {
        final dx = x - cx;
        final dy = y - cy;
        final dist = math.sqrt(dx * dx + dy * dy);
        if (dist <= radius) {
          final falloff = 1 - (dist / radius);
          final alpha = falloff * falloff;
          final pixel = image.getPixel(x, y);
          final r = (pixel.r + (255 - pixel.r) * alpha).clamp(0, 255).toInt();
          final g = (pixel.g + (220 - pixel.g) * alpha * 0.8).clamp(0, 255).toInt();
          final b = (pixel.b + (120 - pixel.b) * alpha * 0.5).clamp(0, 255).toInt();
          image.setPixelRgba(x, y, r, g, b, 255);
        }
      }
    }
  }

  static void _updateBoomerangState(double width, double height, int numBalls, double radius) {
    if (_boomerangBalls.length != numBalls ||
        (width - _boomerangWidth).abs() > 40 ||
        (height - _boomerangHeight).abs() > 40) {
      _boomerangBalls
        ..clear()
        ..addAll(List.generate(numBalls, (_) {
          final x = _random.nextDouble() * (width - 2 * radius) + radius;
          final y = _random.nextDouble() * (height - 2 * radius) + radius;
          final vx = (_random.nextDouble() * 260 - 130);
          final vy = (_random.nextDouble() * 260 - 130);
          return _BoomerangBall(x: x, y: y, vx: vx, vy: vy);
        }));
      _boomerangWidth = width;
      _boomerangHeight = height;
      _numBoomerangBalls = numBalls;
      _lastBoomerangUpdateMs = DateTime.now().millisecondsSinceEpoch;
    }

    final now = DateTime.now().millisecondsSinceEpoch;
    double dt = (now - _lastBoomerangUpdateMs) / 1000.0;
    dt = dt.clamp(0.016, 0.05);
    _lastBoomerangUpdateMs = now;

    final minX = radius;
    final maxX = width - radius;
    final minY = radius;
    final maxY = height - radius;

    for (final ball in _boomerangBalls) {
      ball.x += ball.vx * dt;
      ball.y += ball.vy * dt;

      if (ball.x < minX) {
        ball.x = minX;
        ball.vx = ball.vx.abs();
      } else if (ball.x > maxX) {
        ball.x = maxX;
        ball.vx = -ball.vx.abs();
      }
      if (ball.y < minY) {
        ball.y = minY;
        ball.vy = ball.vy.abs();
      } else if (ball.y > maxY) {
        ball.y = maxY;
        ball.vy = -ball.vy.abs();
      }
    }
  }
  
  /// Procesa imagen ya decodificada
  static img.Image? applyFilterToImage(img.Image image, String filterId) {
    if (filterId == 'none' || filterId == 'caras') return null;

    img.Image working = image;

    // Reducir tamaño para mantener el rendimiento
    if (filterId == 'prewitt' || filterId == 'laplacian') {
      if (working.width > 260) {
        working = img.copyResize(working, width: 260);
      }
    } else {
      if (working.width > 360) {
        working = img.copyResize(working, width: 360);
      }
    }

    switch (filterId) {
      case 'prewitt':
        return applyPrewitt(working);
      case 'laplacian':
        return applyLaplacian(working);
      case 'gaussian':
        return applyGaussian(working);
      case 'box_blur':
        return applyBloxBlur(working);
      case 'ups_logo':
        return applyUpsLogo(working);
      case 'boomerang':
        return applyBoomerang(working);
      default:
        return null;
    }
  }

  /// Procesa imagen con el filtro seleccionado a partir de bytes (usado en capturas)
  static Future<Uint8List?> processImageBytes(Uint8List bytes, String filterId) async {
    if (filterId == 'none' || filterId == 'caras') return null;
    
    try {
      await ensureInitialized();
      final image = img.decodeImage(bytes);
      if (image == null) return null;

      final filtered = applyFilterToImage(image, filterId);
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
      'gaussian': 'Gauss Blur',
      'box_blur': 'Blox Blur',
      'prewitt': 'Prewitt (Bordes)',
      'laplacian': 'Laplace (Bordes)',
      'ups_logo': 'UPS Logo',
      'boomerang': 'Boomerang',
      'caras': 'Caras (Próx.)',
    };
    return filterNames[filter] ?? filter;
  }
}

class _BoomerangBall {
  double x;
  double y;
  double vx;
  double vy;
  _BoomerangBall({
    required this.x,
    required this.y,
    required this.vx,
    required this.vy,
  });
}
