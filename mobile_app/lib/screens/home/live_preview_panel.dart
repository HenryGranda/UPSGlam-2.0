import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';

import 'package:filter_preview_app/utils/local_filters.dart';
import 'package:filter_preview_app/utils/preview_filters.dart';

class LivePreviewPanel extends StatefulWidget {
  final String? filterId;

  const LivePreviewPanel({super.key, required this.filterId});

  @override
  State<LivePreviewPanel> createState() => _LivePreviewPanelState();
}

class _LivePreviewPanelState extends State<LivePreviewPanel> {
  CameraController? _controller;
  bool _initialized = false;
  Uint8List? _snapshot;
  bool _isProcessingFrame = false;
  DateTime _lastFrame = DateTime.fromMillisecondsSinceEpoch(0);
  String? _permissionError;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  @override
  void didUpdateWidget(covariant LivePreviewPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.filterId != widget.filterId) {
      setState(() {
        _snapshot = null;
      });
    }
  }

  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() {
        _permissionError =
            'Permiso de cámara denegado. Activalo para ver la previsualización.';
      });
      return;
    }

    try {
      await LocalFilters.ensureInitialized();
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _permissionError = 'No se encontró ninguna cámara.';
        });
        return;
      }

      _controller = CameraController(
        cameras[0],
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _controller!.initialize();
      await _controller!.setFlashMode(FlashMode.off);
      await _controller!.startImageStream(_processFrame);
      if (!mounted) return;
      setState(() {
        _initialized = true;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _permissionError = 'Error al inicializar la cámara: $e';
      });
    }
  }

  void _processFrame(CameraImage image) {
    if (_controller == null || _isProcessingFrame) return;
    if ((widget.filterId ?? 'none') == 'none') {
      if (_snapshot != null) {
        setState(() {
          _snapshot = null;
        });
      }
      return;
    }

    final now = DateTime.now();
    if (now.difference(_lastFrame).inMilliseconds < 120) {
      return;
    }
    _lastFrame = now;

    _isProcessingFrame = true;
    final currentFilter = widget.filterId ?? 'none';

    try {
      final img.Image rgb = _normalizeOrientation(_cameraImageToImage(image));
      final img.Image? filtered =
          LocalFilters.applyFilterToImage(rgb, currentFilter);
      if (!mounted || currentFilter != (widget.filterId ?? 'none')) {
        _isProcessingFrame = false;
        return;
      }
      if (filtered != null) {
        final bytes = Uint8List.fromList(img.encodeJpg(filtered, quality: 70));
        if (mounted) {
          setState(() {
            _snapshot = bytes;
          });
        }
      }
    } finally {
      _isProcessingFrame = false;
    }
  }

  img.Image _normalizeOrientation(img.Image image) {
    final description = _controller?.description;
    if (description == null) return image;

    int rotation = description.sensorOrientation;
    final bool flipHorizontally =
        description.lensDirection == CameraLensDirection.front;

    img.Image rotated;
    switch (rotation) {
      case 90:
        rotated = img.copyRotate(image, angle: 90);
        break;
      case 270:
        rotated = img.copyRotate(image, angle: -90);
        break;
      case 180:
        rotated = img.copyRotate(image, angle: 180);
        break;
      default:
        rotated = image;
        break;
    }

    if (flipHorizontally) {
      rotated = img.flipHorizontal(rotated);
    }

    return rotated;
  }

  img.Image _cameraImageToImage(CameraImage cameraImage) {
    final width = cameraImage.width;
    final height = cameraImage.height;
    final img.Image image = img.Image(width: width, height: height);

    if (cameraImage.format.group != ImageFormatGroup.yuv420) {
      final plane = cameraImage.planes.first;
      final bytes = plane.bytes;
      int index = 0;
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final r = bytes[index++];
          final g = bytes[index++];
          final b = bytes[index++];
          image.setPixelRgba(x, y, r, g, b, 255);
        }
      }
      return image;
    }

    final planeY = cameraImage.planes[0];
    final planeU = cameraImage.planes[1];
    final planeV = cameraImage.planes[2];
    final bytesPerPixelUV = planeU.bytesPerPixel ?? 1;

    for (int y = 0; y < height; y++) {
      final uvRow = (y >> 1) * planeU.bytesPerRow;
      final yRow = y * planeY.bytesPerRow;
      for (int x = 0; x < width; x++) {
        final uvIndex = uvRow + (x >> 1) * bytesPerPixelUV;
        final yValue = planeY.bytes[yRow + x].toDouble();
        final uValue = planeU.bytes[uvIndex].toDouble() - 128.0;
        final vValue = planeV.bytes[uvIndex].toDouble() - 128.0;

        int r = (yValue + 1.403 * vValue).round();
        int g = (yValue - 0.344 * uValue - 0.714 * vValue).round();
        int b = (yValue + 1.770 * uValue).round();

        final rr = r.clamp(0, 255).toInt();
        final gg = g.clamp(0, 255).toInt();
        final bb = b.clamp(0, 255).toInt();

        image.setPixelRgba(x, y, rr, gg, bb, 255);
      }
    }

    return image;
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_permissionError != null) {
      return Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(24),
          color: Colors.black.withOpacity(0.85),
        ),
        alignment: Alignment.center,
        padding: const EdgeInsets.all(24),
        child: Text(
          _permissionError!,
          style: const TextStyle(color: Colors.white),
          textAlign: TextAlign.center,
        ),
      );
    }

    if (!_initialized || _controller == null) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(24),
      child: Stack(
        fit: StackFit.expand,
        children: [
          _buildPreviewContent(),
          Positioned(
            bottom: 12,
            left: 12,
            right: 12,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.5),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                widget.filterId == null || widget.filterId == 'none'
                    ? 'Selecciona un filtro para la vista previa'
                    : 'Filtro: ${LocalFilters.getFilterDisplayName(widget.filterId!)}',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w500,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPreviewContent() {
    if (_snapshot != null) {
      return Container(
        color: Colors.black,
        alignment: Alignment.center,
        child: FittedBox(
          fit: BoxFit.contain,
          child: Image.memory(
            _snapshot!,
            gaplessPlayback: true,
          ),
        ),
      );
    }

    Widget preview = CameraPreview(_controller!);
    final filter = PreviewFilters.getPreviewColorFilter(widget.filterId ?? 'none');
    if (filter != null) {
      preview = ColorFiltered(
        colorFilter: filter,
        child: preview,
      );
    }

    return Container(
      color: Colors.black,
      alignment: Alignment.center,
      child: FittedBox(
        fit: BoxFit.contain,
        child: SizedBox(
          width: _controller!.value.previewSize?.height ?? 1,
          height: _controller!.value.previewSize?.width ?? 1,
          child: preview,
        ),
      ),
    );
  }
}
