import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;
import '../services/filter_service.dart';
import '../widgets/filter_selector.dart';
import '../widgets/processing_overlay.dart';
import '../utils/local_filters.dart';
import '../utils/preview_filters.dart';
import 'result_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  String _selectedFilter = 'none';
  bool _isProcessing = false;
  String? _permissionError;
  Uint8List? _previewSnapshot;
  bool _isProcessingFrame = false;
  DateTime _lastFrameTime = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    _requestPermissionsAndInitialize();
  }

  Future<void> _requestPermissionsAndInitialize() async {
    final status = await Permission.camera.request();
    
    if (status.isGranted) {
      await _initializeCamera();
    } else if (status.isDenied) {
      setState(() {
        _permissionError = 'Permiso de cámara denegado. Por favor habilítalo en la configuración.';
      });
    } else if (status.isPermanentlyDenied) {
      setState(() {
        _permissionError = 'Permiso de cámara denegado permanentemente. Habilítalo en la configuración de la app.';
      });
    }
  }

  Future<void> _initializeCamera() async {
    try {
      await LocalFilters.ensureInitialized();
      _cameras = await availableCameras();
      if (_cameras!.isEmpty) {
        _showError('No se encontró ninguna cámara');
        return;
      }

      _controller = CameraController(
        _cameras![0],
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _controller!.initialize();
      await _controller!.setFlashMode(FlashMode.off);
      await _startImageStream();
      
      if (mounted) {
        setState(() {
          _isInitialized = true;
        });
      }
    } catch (e) {
      _showError('Error al inicializar la cámara: $e');
    }
  }

  Future<void> _startImageStream() async {
    if (_controller == null || _controller!.value.isStreamingImages) return;
    await _controller!.startImageStream(_processCameraImage);
  }

  Future<void> _stopImageStream() async {
    if (_controller == null || !_controller!.value.isStreamingImages) return;
    await _controller!.stopImageStream();
  }

  void _processCameraImage(CameraImage cameraImage) {
    if (_isProcessing || _controller == null) return;
    if (_selectedFilter == 'none' || _selectedFilter == 'caras') {
      if (_previewSnapshot != null) {
        setState(() {
          _previewSnapshot = null;
        });
      }
      return;
    }
    if (_isProcessingFrame) return;
    final now = DateTime.now();
    if (now.difference(_lastFrameTime).inMilliseconds < 120) {
      return;
    }
    _lastFrameTime = now;

    _isProcessingFrame = true;
    final currentFilter = _selectedFilter;

    try {
      final img.Image rgb = _cameraImageToImage(cameraImage);
      final img.Image? filtered = LocalFilters.applyFilterToImage(rgb, currentFilter);
      if (!mounted || currentFilter != _selectedFilter) {
        _isProcessingFrame = false;
        return;
      }
      if (filtered != null) {
        final bytes = Uint8List.fromList(img.encodeJpg(filtered, quality: 70));
        if (mounted) {
          setState(() {
            _previewSnapshot = bytes;
          });
        }
      }
    } catch (e) {
      // Ignorar errores de conversión individuales
    } finally {
      _isProcessingFrame = false;
    }
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

  void _showError(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  /// Se llama cuando cambia el filtro seleccionado
  void _onFilterChanged(String newFilter) {
    setState(() {
      _selectedFilter = newFilter;
      _previewSnapshot = null;
    });
  }

  Future<void> _captureAndProcess() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      _showError('La cámara no está lista');
      return;
    }

    if (_selectedFilter == 'none' || _selectedFilter == 'caras') {
      _showError('Por favor selecciona un filtro');
      return;
    }

    setState(() {
      _isProcessing = true;
    });

    final wasStreaming = _controller!.value.isStreamingImages;

    try {
      if (wasStreaming) {
        await _stopImageStream();
      }
      // Capture original image
      final XFile image = await _controller!.takePicture();
      final Uint8List imageBytes = await image.readAsBytes();

      // Send to backend for PyCUDA processing
      final Uint8List? processedImage = await FilterService.applyFilter(
        imageBytes,
        _selectedFilter,
      );

      if (processedImage != null && mounted) {
        // Navigate to result screen
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (context) => ResultScreen(
              originalImage: imageBytes,
              processedImage: processedImage,
              filterName: _selectedFilter,
            ),
          ),
        );
      } else {
        _showError('Error al procesar la imagen');
      }
    } catch (e) {
      _showError('Error al capturar la imagen: $e');
    } finally {
      if (wasStreaming) {
        await _startImageStream();
      }
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Show permission error
    if (_permissionError != null) {
      return Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(
                  Icons.camera_alt_outlined,
                  size: 80,
                  color: Colors.white54,
                ),
                const SizedBox(height: 24),
                Text(
                  _permissionError!,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 24),
                ElevatedButton(
                  onPressed: () => openAppSettings(),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFF2A900),
                  ),
                  child: const Text('Abrir Configuración'),
                ),
              ],
            ),
          ),
        ),
      );
    }

    if (!_isInitialized) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Camera Preview with filter applied
          Positioned.fill(
            child: _buildFilteredPreview(),
          ),

          // Filter info overlay (top)
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.6),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  _selectedFilter == 'none'
                      ? 'Selecciona un filtro'
                      : 'Filtro: ${_getFilterDisplayName(_selectedFilter)}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ),
          ),

          // Filter Selector (bottom)
          Positioned(
            bottom: 120,
            left: 0,
            right: 0,
            child: FilterSelector(
              selectedFilter: _selectedFilter,
              onFilterSelected: _onFilterChanged,
            ),
          ),

          // Capture Button
          Positioned(
            bottom: 32,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: _isProcessing ? null : _captureAndProcess,
                child: Container(
                  width: 70,
                  height: 70,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _selectedFilter == 'none'
                        ? Colors.grey
                        : const Color(0xFFF2A900),
                    border: Border.all(
                      color: Colors.white,
                      width: 4,
                    ),
                  ),
                  child: _isProcessing
                      ? const Padding(
                          padding: EdgeInsets.all(12.0),
                          child: CircularProgressIndicator(
                            color: Colors.white,
                            strokeWidth: 3,
                          ),
                        )
                      : const Icon(
                          Icons.camera_alt,
                          color: Colors.white,
                          size: 32,
                        ),
                ),
              ),
            ),
          ),

          // Processing Overlay
          if (_isProcessing)
            const ProcessingOverlay(),
        ],
      ),
    );
  }
//snapshot procesado
  Widget _buildFilteredPreview() {
    // Si hay snapshot procesado, mostrarlo
    if (_previewSnapshot != null) {
      return Image.memory(
        _previewSnapshot!,
        fit: BoxFit.cover,
        gaplessPlayback: true,
      );
    }
    
    // Sino, mostrar cámara normal con filtro rápido visual
    Widget preview = CameraPreview(_controller!);
    final colorFilter = PreviewFilters.getPreviewColorFilter(_selectedFilter);
    if (colorFilter != null) {
      preview = ColorFiltered(
        colorFilter: colorFilter,
        child: preview,
      );
    }
    return preview;
  }

  String _getFilterDisplayName(String filter) {
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
