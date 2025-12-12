// lib/screens/home/create_post_view.dart
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../models/current_user.dart';
import '../../models/post_model.dart';
import '../../services/post_service.dart';
import 'live_preview_panel.dart';

class CreatePostView extends StatefulWidget {
  final void Function(PostModel) onPublish;
  final VoidCallback onBack;
  final CurrentUser? currentUser;

  const CreatePostView({
    super.key,
    required this.onPublish,
    required this.onBack,
    this.currentUser,
  });

  @override
  State<CreatePostView> createState() => _CreatePostViewState();
}

class _CreatePostViewState extends State<CreatePostView> {
  final TextEditingController _descriptionController = TextEditingController();
  final ImagePicker _picker = ImagePicker();

  String? _previewImagePath; // archivo local elegido
  bool _publishing = false;
  bool _previewEnabled = false;

  /// Filtros tal cual los espera el backend
  final List<String> _filters = [
    'gaussian',
    'box_blur',
    'prewitt',
    'laplacian',
    'ups_logo',
    'ups_color',
  ];

  String? _selectedFilter; // este string va directo al backend

  bool get _hasRealImage => _previewImagePath != null;
  bool get _canPublish =>
      _hasRealImage && _descriptionController.text.trim().isNotEmpty;
  bool get _canCapture => !_previewEnabled;

  // --- Cámara ---
  Future<void> _onTakePhoto() async {
    if (!_canCapture) return;
    final picked =
        await _picker.pickImage(source: ImageSource.camera, imageQuality: 80);
    if (picked == null) return;
    setState(() {
      _previewImagePath = picked.path;
    });
  }

  // --- Galería ---
  Future<void> _onPickFromGallery() async {
    if (!_canCapture) return;
    final picked =
        await _picker.pickImage(source: ImageSource.gallery, imageQuality: 80);
    if (picked == null) return;
    setState(() {
      _previewImagePath = picked.path;
    });
  }

  Future<void> _publish() async {
    if (!_canPublish || _previewImagePath == null) return;

    try {
      setState(() {
        _publishing = true;
      });

      final caption = _descriptionController.text.trim();
      final user = widget.currentUser;

      await PostService.instance.createPost(
        imageFile: File(_previewImagePath!),
        caption: caption,
        filter: _selectedFilter,
        username: user?.username,
        userPhotoUrl: user?.photoUrl,
      );

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Publicación creada correctamente")),
      );

      _descriptionController.clear();
      setState(() {
        _previewImagePath = null;
        _selectedFilter = null;
      });

      widget.onBack(); // vuelve al feed, donde recargas
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error al publicar: $e")),
      );
    } finally {
      setState(() {
        _publishing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return GestureDetector(
      // swipe hacia abajo para salir
      onVerticalDragEnd: (details) {
        if (details.primaryVelocity != null &&
            details.primaryVelocity! > 600) {
          widget.onBack();
        }
      },
      child: Column(
        children: [
          // HEADER
          SafeArea(
            bottom: false,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              child: Row(
                children: [
                  IconButton(
                    onPressed: widget.onBack,
                    icon: const Icon(Icons.arrow_back),
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    'Crear publicación',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // CONTENIDO
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // VISTA PREVIA
                  AspectRatio(
                    aspectRatio: 3 / 4,
                    child: _previewEnabled
                        ? LivePreviewPanel(filterId: _selectedFilter)
                        : Container(
                            decoration: BoxDecoration(
                              color: Colors.grey[200],
                              borderRadius: BorderRadius.circular(24),
                            ),
                            clipBehavior: Clip.antiAlias,
                            child: _previewImagePath == null
                                ? const Center(
                                    child: Icon(Icons.image, size: 40),
                                  )
                                : Image.file(
                                    File(_previewImagePath!),
                                    fit: BoxFit.cover,
                                  ),
                          ),
                  ),

                  const SizedBox(height: 12),

                  // TOGGLE PREVIEW
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Modo previsualización (sólo demostración)'),
                    subtitle: const Text(
                      'Desactiva para capturar foto y enviarla al backend.',
                    ),
                    value: _previewEnabled,
                    onChanged: (value) {
                      setState(() {
                        _previewEnabled = value;
                        if (value) {
                          _previewImagePath = null;
                        }
                      });
                    },
                  ),

                  const SizedBox(height: 16),

                  // BOTONES: TOMAR FOTO / GALERÍA
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _canCapture ? _onTakePhoto : null,
                          icon: const Icon(Icons.camera_alt_outlined),
                          label: const Text('Tomar foto'),
                          style: OutlinedButton.styleFrom(
                            padding:
                                const EdgeInsets.symmetric(vertical: 14),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _canCapture ? _onPickFromGallery : null,
                          icon: const Icon(Icons.photo_library_outlined),
                          label: const Text('Galería'),
                          style: OutlinedButton.styleFrom(
                            padding:
                                const EdgeInsets.symmetric(vertical: 14),
                          ),
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 20),

                  // FILTROS
                  Text(
                    'Filtros GPU (backend)',
                    style: theme.textTheme.titleSmall!
                        .copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: _filters.map((code) {
                      final selected = _selectedFilter == code;
                      return ChoiceChip(
                        label: Text(code),
                        selected: selected,
                        onSelected: (_) {
                          setState(() {
                            _selectedFilter = selected ? null : code;
                          });
                        },
                        selectedColor: theme.colorScheme.primary,
                        labelStyle: TextStyle(
                          color: selected
                              ? theme.colorScheme.onPrimary
                              : theme.colorScheme.onSurface,
                        ),
                      );
                    }).toList(),
                  ),

                  const SizedBox(height: 20),

                  // DESCRIPCIÓN
                  Text(
                    'Descripción',
                    style: theme.textTheme.titleSmall!
                        .copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 6),
                  TextField(
                    controller: _descriptionController,
                    maxLines: 3,
                    decoration: InputDecoration(
                      hintText: 'Escribe una descripción…',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // BOTÓN PUBLICAR
          Padding(
            padding: const EdgeInsets.only(
                left: 16, right: 16, bottom: 16, top: 4),
            child: SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: _canPublish && !_publishing ? _publish : null,
                child: Padding(
                  padding:
                      const EdgeInsets.symmetric(vertical: 14),
                  child: _publishing
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Text(
                          'Publicar',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
