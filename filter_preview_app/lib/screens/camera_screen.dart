import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/filter_service.dart';
import '../widgets/filter_selector.dart';
import '../widgets/processing_overlay.dart';
import '../utils/local_filters.dart';
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
  Timer? _snapshotTimer;
  bool _isUpdatingSnapshot = false;

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
      _cameras = await availableCameras();
      if (_cameras!.isEmpty) {
        _showError('No se encontró ninguna cámara');
        return;
      }

      _controller = CameraController(
        _cameras![0],
        ResolutionPreset.high,
        enableAudio: false,
      );

      await _controller!.initialize();
      
      if (mounted) {
        setState(() {
          _isInitialized = true;
        });
      }
    } catch (e) {
      _showError('Error al inicializar la cámara: $e');
    }
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

  /// Actualiza snapshot con filtro procesado
  Future<void> _updateFilterSnapshot() async {
    if (_isUpdatingSnapshot || _controller == null || !_controller!.value.isInitialized) {
      return;
    }
    
    if (_selectedFilter == 'none') {
      setState(() {
        _previewSnapshot = null;
      });
      return;
    }
    
    _isUpdatingSnapshot = true;
    
    try {
      final image = await _controller!.takePicture();
      final bytes = await image.readAsBytes();
      
      // Importar local_filters para procesamiento
      final filtered = await LocalFilters.processImageBytes(bytes, _selectedFilter);
      
      if (mounted && filtered != null) {
        setState(() {
          _previewSnapshot = filtered;
        });
      }
    } catch (e) {
      print('Error updating snapshot: $e');
    } finally {
      _isUpdatingSnapshot = false;
    }
  }
  
  /// Inicia/detiene el timer de snapshots según el filtro
  void _manageSnapshotTimer() {
    _snapshotTimer?.cancel();
    
    if (_selectedFilter != 'none') {
      // Actualizar inmediatamente
      _updateFilterSnapshot();
      
      // Configurar timer para actualizar cada 500ms
      _snapshotTimer = Timer.periodic(
        const Duration(milliseconds: 500),
        (_) => _updateFilterSnapshot(),
      );
    } else {
      setState(() {
        _previewSnapshot = null;
      });
    }
  }
  
  /// Se llama cuando cambia el filtro seleccionado
  void _onFilterChanged(String newFilter) {
    setState(() {
      _selectedFilter = newFilter;
    });
    _manageSnapshotTimer();
  }

  Future<void> _captureAndProcess() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      _showError('La cámara no está lista');
      return;
    }

    if (_selectedFilter == 'none') {
      _showError('Por favor selecciona un filtro');
      return;
    }

    setState(() {
      _isProcessing = true;
    });

    try {
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
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _snapshotTimer?.cancel();
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
    
    // Sino, mostrar cámara normal
    return CameraPreview(_controller!);
  }

  String _getFilterDisplayName(String filter) {
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