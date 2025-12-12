import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'dart:typed_data';
import 'dart:math' as math;

/// Filtros locales que simulan los filtros oficiales del backend CUDA.
/// Útiles únicamente para previsualizar efectos en tiempo real.
class LocalFilters {
  static bool _assetsLoaded = false;
  static img.Image? _sonrisaTexture;
  static img.Image? _upsOverlayTexture;
  static img.Image? _faceMaskTexture;
  static final Map<int, img.Image> _resizedTextureCache = {};
  static final Map<String, img.Image> _upsOverlayCache = {};
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

      final overlayData = await rootBundle.load('assets/filtro_don_bosco.png');
      final overlayBytes = overlayData.buffer.asUint8List();
      _upsOverlayTexture = img.decodeImage(overlayBytes);

      final faceMaskData = await rootBundle.load('assets/face_mask.png');
      final faceMaskBytes = faceMaskData.buffer.asUint8List();
      _faceMaskTexture = img.decodeImage(faceMaskBytes);
    } catch (_) {
      _sonrisaTexture = null;
      _upsOverlayTexture = null;
      _faceMaskTexture = null;
    } finally {
      _assetsLoaded = true;
    }
  }

  /// Aplica filtro Prewitt OPTIMIZADO para tiempo real
  static img.Image applyPrewitt(img.Image image, {double gain = 3.0}) {
    final gray = img.grayscale(image);
    final w = gray.width;
    final h = gray.height;
    final result = img.Image(width: w, height: h);

    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        final gx =
            -gray.getPixel(x - 1, y - 1).r.toDouble() +
                gray.getPixel(x + 1, y - 1).r.toDouble() +
                -gray.getPixel(x - 1, y).r.toDouble() +
                gray.getPixel(x + 1, y).r.toDouble() +
                -gray.getPixel(x - 1, y + 1).r.toDouble() +
                gray.getPixel(x + 1, y + 1).r.toDouble();

        final gy =
            -gray.getPixel(x - 1, y - 1).r.toDouble() +
                -gray.getPixel(x, y - 1).r.toDouble() +
                -gray.getPixel(x + 1, y - 1).r.toDouble() +
                gray.getPixel(x - 1, y + 1).r.toDouble() +
                gray.getPixel(x, y + 1).r.toDouble() +
                gray.getPixel(x + 1, y + 1).r.toDouble();

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

  /// Aplica Gaussian Blur (de la librería)
  static img.Image applyGaussian(img.Image image) {
    return img.gaussianBlur(image, radius: 4);
  }

  /// Aplica efecto Blox Blur con promedio suave
  static img.Image applyBloxBlur(
    img.Image image, {
    int radius = 3,
    int passes = 1,
  }) {
    radius = radius.clamp(1, 12);
    passes = passes.clamp(1, 2);

    img.Image src = img.Image.from(image);
    final img.Image temp = img.Image(width: image.width, height: image.height);

    for (int i = 0; i < passes; i++) {
      _boxBlurHorizontal(src, temp, radius);
      _boxBlurVertical(temp, src, radius);
    }

    return src;
  }

  static void _boxBlurHorizontal(img.Image src, img.Image dst, int radius) {
    final width = src.width;
    final height = src.height;
    final window = radius * 2 + 1;

    for (int y = 0; y < height; y++) {
      double rSum = 0, gSum = 0, bSum = 0, aSum = 0;

      for (int ix = -radius; ix <= radius; ix++) {
        final sx = ix.clamp(0, width - 1);
        final pixel = src.getPixel(sx, y);
        rSum += pixel.r;
        gSum += pixel.g;
        bSum += pixel.b;
        aSum += pixel.a;
      }

      for (int x = 0; x < width; x++) {
        dst.setPixelRgba(
          x,
          y,
          (rSum / window).round().clamp(0, 255),
          (gSum / window).round().clamp(0, 255),
          (bSum / window).round().clamp(0, 255),
          (aSum / window).round().clamp(0, 255),
        );

        final removeX = (x - radius).clamp(0, width - 1);
        final addX = (x + radius + 1).clamp(0, width - 1);
        final removePixel = src.getPixel(removeX, y);
        final addPixel = src.getPixel(addX, y);

        rSum += addPixel.r - removePixel.r;
        gSum += addPixel.g - removePixel.g;
        bSum += addPixel.b - removePixel.b;
        aSum += addPixel.a - removePixel.a;
      }
    }
  }

  static void _boxBlurVertical(img.Image src, img.Image dst, int radius) {
    final width = src.width;
    final height = src.height;
    final window = radius * 2 + 1;

    for (int x = 0; x < width; x++) {
      double rSum = 0, gSum = 0, bSum = 0, aSum = 0;

      for (int iy = -radius; iy <= radius; iy++) {
        final sy = iy.clamp(0, height - 1);
        final pixel = src.getPixel(x, sy);
        rSum += pixel.r;
        gSum += pixel.g;
        bSum += pixel.b;
        aSum += pixel.a;
      }

      for (int y = 0; y < height; y++) {
        dst.setPixelRgba(
          x,
          y,
          (rSum / window).round().clamp(0, 255),
          (gSum / window).round().clamp(0, 255),
          (bSum / window).round().clamp(0, 255),
          (aSum / window).round().clamp(0, 255),
        );

        final removeY = (y - radius).clamp(0, height - 1);
        final addY = (y + radius + 1).clamp(0, height - 1);
        final removePixel = src.getPixel(x, removeY);
        final addPixel = src.getPixel(x, addY);

        rSum += addPixel.r - removePixel.r;
        gSum += addPixel.g - removePixel.g;
        bSum += addPixel.b - removePixel.b;
        aSum += addPixel.a - removePixel.a;
      }
    }
  }

  /// Versión clásica de box blur (respaldo)
  static img.Image applyBoxBlur(img.Image image) {
    return img.convolution(
      image,
      filter: [
        1 / 9, 1 / 9, 1 / 9,
        1 / 9, 1 / 9, 1 / 9,
        1 / 9, 1 / 9, 1 / 9,
      ],
    );
  }

  /// Aplica efecto Don Bosco (aura dorada + overlay UPS)
  static img.Image applyUpsLogo(img.Image image) {
    final width = image.width;
    final height = image.height;
    img.Image? overlayTexture;
    if (_upsOverlayTexture != null) {
      overlayTexture = _getUpsOverlayResized(
        _upsOverlayTexture!,
        width,
        height,
      );
    }

    final baseOverlay = overlayTexture ?? image;
    final result = img.Image.from(baseOverlay);
    final time = DateTime.now().millisecondsSinceEpoch / 1000.0;

    const brown = [58, 44, 26];
    const gold = [242, 169, 0];

    const threshold = 0.35;
    const waveAmplitude = 0.02;
    const waveFrequency = 14.0;
    const glowStrength = 2.5;
    const auraBlend = 0.65;
    const haloStrength = 0.45;
    const haloFalloff = 0.06;
    const overlayTintStrength = 0.3;
    const oscillationPx = 8.0;
    const oscillationSpeed = 1.4;

    final overlayWidth = (width * 0.33).round();
    final overlayHeight = (height * 0.33).round();
    final overlayLeft = (width * 0.60).round();
    final overlayTop = (height * 0.40).round();
    final overlayRight = overlayLeft + overlayWidth;
    final overlayBottom = overlayTop + overlayHeight;

    final cameraRegion = img.copyResize(
      image,
      width: overlayWidth,
      height: overlayHeight,
    );

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final overlayPixel = baseOverlay.getPixel(x, y);
        double fr = overlayPixel.r / 255.0;
        double fg = overlayPixel.g / 255.0;
        double fb = overlayPixel.b / 255.0;
        double lumValue =
            (0.2126 * overlayPixel.r +
                    0.7152 * overlayPixel.g +
                    0.0722 * overlayPixel.b) /
                255.0;
        double neigh = 0;
        for (int oy = -1; oy <= 1; oy++) {
          for (int ox = -1; ox <= 1; ox++) {
            final nx = (x + ox).clamp(0, width - 1);
            final ny = (y + oy).clamp(0, height - 1);
            final np = baseOverlay.getPixel(nx, ny);
            neigh += (0.2126 * np.r +
                    0.7152 * np.g +
                    0.0722 * np.b) /
                255.0;
          }
        }
        neigh /= 9.0;

        final mask = lumValue > threshold ? 1.0 : 0.0;
        final glowVal = (lumValue - neigh).abs() * glowStrength;
        final auraMask = (mask + glowVal).clamp(0.0, 1.0);

        final nxNorm = x / (width - 1);
        final nyNorm = y / (height - 1);
        final wave =
            math.sin((nxNorm + nyNorm) * waveFrequency + time) * waveAmplitude;

        final sx = ((nxNorm + wave).clamp(0.0, 1.0) * (width - 1)).round();
        final sy = ((nyNorm + wave).clamp(0.0, 1.0) * (height - 1)).round();
        final warpPixel = baseOverlay.getPixel(sx, sy);

        final pal =
            0.5 * (math.sin(time + nxNorm * 10 - nyNorm * 10) + 1.0);
        final auraR = (brown[0] * (1 - pal) + gold[0] * pal) / 255.0;
        final auraG = (brown[1] * (1 - pal) + gold[1] * pal) / 255.0;
        final auraB = (brown[2] * (1 - pal) + gold[2] * pal) / 255.0;

        final aA = (auraMask * auraBlend).clamp(0.0, 1.0);
        fr = warpPixel.r / 255.0 * (1.0 - aA) + auraR * aA;
        fg = warpPixel.g / 255.0 * (1.0 - aA) + auraG * aA;
        fb = warpPixel.b / 255.0 * (1.0 - aA) + auraB * aA;

        final dynTop =
            overlayTop + math.sin(time * oscillationSpeed) * oscillationPx;
        final dynBottom = dynTop + overlayHeight;
        final dynLeft = overlayLeft.toDouble();
        final dynRight = dynLeft + overlayWidth;

        double dxh = 0.0;
        if (x < dynLeft) dxh = dynLeft - x;
        else if (x > dynRight) dxh = x - dynRight;

        double dyh = 0.0;
        if (y < dynTop) dyh = dynTop - y;
        else if (y > dynBottom) dyh = y - dynBottom;

        final dist = math.sqrt(dxh * dxh + dyh * dyh);
        final halo = math.exp(-dist * haloFalloff) * haloStrength;

        fr = (fr + (gold[0] / 255.0) * halo).clamp(0.0, 1.0);
        fg = (fg + (gold[1] / 255.0) * halo).clamp(0.0, 1.0);
        fb = (fb + (gold[2] / 255.0) * halo).clamp(0.0, 1.0);

        if (x >= overlayLeft &&
            x < overlayRight &&
            y >= overlayTop &&
            y < overlayBottom) {
          final localX = (x - overlayLeft).clamp(0, overlayWidth - 1);
          final localY = (y - overlayTop).clamp(0, overlayHeight - 1);
          final camPixel = cameraRegion.getPixel(localX, localY);
          fr = camPixel.r / 255.0;
          fg = camPixel.g / 255.0;
          fb = camPixel.b / 255.0;
        }

        result.setPixelRgba(
          x,
          y,
          (fr * 255).clamp(0, 255).toInt(),
          (fg * 255).clamp(0, 255).toInt(),
          (fb * 255).clamp(0, 255).toInt(),
          255,
        );
      }
    }

    return result;
  }

  /// Aplica efecto Boomerang simulando las pelotas luminosas del filtro CUDA
  static img.Image applyBoomerang(img.Image image, {int numOrbs = 3}) {
    final result = img.Image.from(image);
    final minSide = math.min(result.width, result.height).toDouble();
    final radius = math.max(14, math.min(90, (minSide * 0.07))).toDouble();

    _updateBoomerangState(
      result.width.toDouble(),
      result.height.toDouble(),
      numOrbs,
      radius,
    );

    for (final pixel in result) {
      pixel
        ..r = (pixel.r * 1.05 + 10).clamp(0, 255).toInt()
        ..g = (pixel.g * 1.03 + 5).clamp(0, 255).toInt()
        ..b = (pixel.b * 1.08 + 8).clamp(0, 255).toInt();
    }

    for (final ball in _boomerangBalls) {
      final overlay = _getOrbTexture(radius);

      final dstX = (ball.x - radius).round();
      final dstY = (ball.y - radius).round();

      if (overlay != null) {
        img.compositeImage(
          result,
          overlay,
          dstX: dstX,
          dstY: dstY,
          blend: img.BlendMode.alpha,
        );
      } else {
        _paintFallbackOrb(result, ball.x, ball.y, radius);
      }

      if (overlay != null) {
        final trailX = (ball.x - ball.vx * 0.02 - radius * 0.7).round();
        final trailY = (ball.y - ball.vy * 0.02 - radius * 0.7).round();
        final smaller = _getOrbTexture(radius * 0.75);
        if (smaller != null) {
          img.compositeImage(
            result,
            smaller,
            dstX: trailX,
            dstY: trailY,
            blend: img.BlendMode.alpha,
          );
        }
      }
    }

    return result;
  }

  /// Aplica overlay de máscara facial simple para previsualización
  static img.Image applyFaceMask(img.Image image) {
    final mask = _faceMaskTexture;
    if (mask == null) {
      return image;
    }

    final result = img.Image.from(image);
    final targetWidth = (result.width * 0.6).clamp(12.0, result.width.toDouble());
    final aspect = mask.height / mask.width;
    final targetHeight =
        (targetWidth * aspect).clamp(8.0, result.height.toDouble());
    double dstX = (result.width - targetWidth) / 2;
    double dstY = result.height * 0.18;
    dstX = dstX.clamp(0.0, result.width - targetWidth);
    dstY = dstY.clamp(0.0, result.height - targetHeight);

    final resizedMask = img.copyResize(
      mask,
      width: targetWidth.round(),
      height: targetHeight.round(),
      interpolation: img.Interpolation.cubic,
    );

    img.compositeImage(
      result,
      resizedMask,
      dstX: dstX.round(),
      dstY: dstY.round(),
      blend: img.BlendMode.alpha,
    );

    return result;
  }

  static img.Image? _getOrbTexture(double radius) {
    final size = (radius * 2).round().clamp(4, 512);
    if (_resizedTextureCache.containsKey(size)) {
      return _resizedTextureCache[size];
    }
    final orb = _buildOrbTexture(size);
    if (_resizedTextureCache.length > 6) {
      _resizedTextureCache.remove(_resizedTextureCache.keys.first);
    }
    _resizedTextureCache[size] = orb;
    return orb;
  }

  static img.Image _buildOrbTexture(int size) {
    final orb = img.Image(width: size, height: size);
    final center = (size - 1) / 2.0;
    final radius = center - 1.0;
    final strokeWidth = 3.0;

    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        final dx = x - center;
        final dy = y - center;
        final dist = math.sqrt(dx * dx + dy * dy);
        if (dist > radius) {
          orb.setPixelRgba(x, y, 0, 0, 0, 0);
          continue;
        }

        if (dist >= radius - strokeWidth) {
          orb.setPixelRgba(x, y, 255, 255, 255, 255);
        } else {
          orb.setPixelRgba(x, y, 255, 255, 255, 200);
        }
      }
    }

    final texture = _sonrisaTexture;
    if (texture != null) {
      final innerSize = (size * 0.6).round().clamp(4, size - 4);
      final resized = img.copyResize(
        texture,
        width: innerSize,
        height: innerSize,
        interpolation: img.Interpolation.cubic,
      );
      final dstX = ((size - innerSize) / 2).round();
      final dstY = ((size - innerSize) / 2).round();
      img.compositeImage(
        orb,
        resized,
        dstX: dstX,
        dstY: dstY,
        blend: img.BlendMode.alpha,
      );
    }

    return orb;
  }

  static img.Image? _getUpsOverlayResized(
    img.Image base,
    int width,
    int height,
  ) {
    final key = '${width}x$height';
    if (_upsOverlayCache.containsKey(key)) {
      return _upsOverlayCache[key];
    }
    final resized = img.copyResize(
      base,
      width: width,
      height: height,
      interpolation: img.Interpolation.cubic,
    );
    if (_upsOverlayCache.length > 4) {
      _upsOverlayCache.remove(_upsOverlayCache.keys.first);
    }
    _upsOverlayCache[key] = resized;
    return resized;
  }

  static void _paintFallbackOrb(
    img.Image image,
    double cx,
    double cy,
    double radius,
  ) {
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
          final r =
              (pixel.r + (255 - pixel.r) * alpha).clamp(0, 255).toInt();
          final g =
              (pixel.g + (220 - pixel.g) * alpha * 0.8).clamp(0, 255).toInt();
          final b =
              (pixel.b + (120 - pixel.b) * alpha * 0.5).clamp(0, 255).toInt();
          image.setPixelRgba(x, y, r, g, b, 255);
        }
      }
    }
  }

  static void _updateBoomerangState(
    double width,
    double height,
    int numBalls,
    double radius,
  ) {
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
    if (filterId == 'none') return null;

    img.Image working = image;

    const int maxWidthEdges = 260;
    const int maxWidthBlur = 420;
    const int maxWidthFx = 420;

    // 🔹 Siempre reducimos para PREVIEW en vivo (según tipo de filtro)
    if (filterId == 'prewitt' || filterId == 'laplacian') {
      if (working.width > maxWidthEdges) {
        working = img.copyResize(working, width: maxWidthEdges);
      }
    } else if (filterId == 'gaussian' || filterId == 'box_blur') {
      if (working.width > maxWidthBlur) {
        working = img.copyResize(working, width: maxWidthBlur);
      }
    } else if (filterId == 'ups_logo' ||
        filterId == 'boomerang' ||
        filterId == 'caras') {
      if (working.width > maxWidthFx) {
        working = img.copyResize(working, width: maxWidthFx);
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
        return applyBloxBlur(working, radius: 3, passes: 1);
      case 'ups_logo':
        return applyUpsLogo(working);
      case 'boomerang':
        return applyBoomerang(working);
      case 'caras':
        return applyFaceMask(working);
      default:
        return null;
    }
  }

  /// Procesa imagen con el filtro seleccionado a partir de bytes (usado en capturas)
  static Future<Uint8List?> processImageBytes(
    Uint8List bytes,
    String filterId,
  ) async {
    if (filterId == 'none') return null;

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
      'caras': 'Caras',
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
