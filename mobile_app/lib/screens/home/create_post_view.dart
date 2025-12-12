// lib/screens/home/create_post_view.dart
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:audioplayers/audioplayers.dart';

import '../../models/current_user.dart';
import '../../models/post_model.dart';
import '../../services/post_service.dart';
import '../../services/filter_service.dart';
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
  final AudioPlayer _audioPlayer = AudioPlayer();

  String? _originalImagePath; // imagen original sin filtro
  String? _previewImagePath; // imagen a mostrar (puede ser original o filtrada)
  bool _publishing = false;
  bool _previewEnabled = false;
  bool _applyingFilter = false; // indica si se está aplicando un filtro

  // 🚫 Palabras prohibidas (censura)
  final List<String> _bannedWords = [
    'messi',
    'barcelona',
    'visca barca',
    'barça',
    'hitler',
    'nazi',
    'puto',
    'pendejo',
    // Agrega más palabras según necesites
  ];

  @override
  void initState() {
    super.initState();
    // Escuchar cambios en el texto para actualizar el estado del botón
    _descriptionController.addListener(() {
      setState(() {
        // El estado se actualiza cuando el texto cambia
      });
    });
  }

  @override
  void dispose() {
    _descriptionController.dispose();
    _audioPlayer.dispose();
    super.dispose();
  }

  /// Filtros tal cual los espera el backend + nombre amigable
  final List<Map<String, String>> _filterOptions = [
    {'code': 'gaussian', 'label': 'Gaussian Blur'},
    {'code': 'box_blur', 'label': 'Box Blur'},
    {'code': 'prewitt', 'label': 'Prewitt'},
    {'code': 'laplacian', 'label': 'Laplacian'},
    {'code': 'ups_logo', 'label': 'UPS Logo'},
    {'code': 'boomerang', 'label': 'Boomerang'},
    {'code': 'cr7', 'label': 'CR7 Mask'},
  ];

  String? _selectedFilter; // filtro seleccionado actualmente
  String? _selectedAudio; // audio seleccionado

  // Audios disponibles en assets
  final List<Map<String, String>> _audioOptions = [
    {'file': 'Siuu.mp3', 'label': '⚽ Siuuu - CR7'},
    {'file': 'cancioncr7_FRPRUlK5.mp3', 'label': '🎵 Canción CR7'},
    {'file': 'vamonosamartecr7_p9FG2LGY.mp3', 'label': '❤️ Vámonos a Amar'},
    {'file': 'waka_IEJmUo0a.mp3', 'label': '⚽ Waka Waka'},
    {'file': 'kuduro_isTLScH6.mp3', 'label': '🔥 Kuduro'},
    {'file': 'nadiedominar_XcY3jlnG.mp3', 'label': '🏆 Nadie Dominar'},
  ];

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
      _originalImagePath = picked.path;
      _previewImagePath = picked.path;
      _selectedFilter = null; // reset filtro al tomar nueva foto
      // No resetear _selectedAudio, puede mantenerlo
    });
  }

  // --- Galería ---
  Future<void> _onPickFromGallery() async {
    if (!_canCapture) return;
    final picked =
        await _picker.pickImage(source: ImageSource.gallery, imageQuality: 80);
    if (picked == null) return;
    setState(() {
      _originalImagePath = picked.path;
      _previewImagePath = picked.path;
      _selectedFilter = null; // reset filtro al elegir nueva foto
      // No resetear _selectedAudio, puede mantenerlo
    });
  }

  // --- Aplicar Filtro ---
  Future<void> _applyFilter(String filterCode) async {
    if (_originalImagePath == null) return;
    if (_applyingFilter) return; // evitar múltiples peticiones simultáneas

    setState(() {
      _applyingFilter = true;
    });

    try {
      // Aplicar filtro a través del API Gateway -> CUDA Backend
      final filteredPath = await FilterService.instance.applyFilter(
        imageFile: File(_originalImagePath!),
        filterName: filterCode,
      );

      // Actualizar la preview con la imagen filtrada
      setState(() {
        _previewImagePath = filteredPath;
        _selectedFilter = filterCode;
      });

      // Reproducir el audio "Siuu" si es el filtro CR7
      if (filterCode == 'cr7') {
        try {
          await _audioPlayer.play(AssetSource('audios/Siuu.mp3'));
        } catch (e) {
          debugPrint('Error al reproducir audio CR7: $e');
        }
      }

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Filtro "$filterCode" aplicado correctamente'),
          duration: const Duration(seconds: 2),
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error al aplicar filtro: $e'),
          duration: const Duration(seconds: 3),
          backgroundColor: Colors.red,
        ),
      );
      // Revertir a imagen original si falla
      setState(() {
        _previewImagePath = _originalImagePath;
        _selectedFilter = null;
      });
    } finally {
      setState(() {
        _applyingFilter = false;
      });
    }
  }

  // --- Remover Filtro ---
  void _removeFilter() {
    if (_originalImagePath == null) return;
    setState(() {
      _previewImagePath = _originalImagePath;
      _selectedFilter = null;
    });
  }

  // Validar si el texto contiene palabras prohibidas
  bool _containsBannedWords(String text) {
    final lowerText = text.toLowerCase();
    for (var word in _bannedWords) {
      if (lowerText.contains(word.toLowerCase())) {
        return true;
      }
    }
    return false;
  }

  // Obtener la primera palabra prohibida encontrada
  String? _getFirstBannedWord(String text) {
    final lowerText = text.toLowerCase();
    for (var word in _bannedWords) {
      if (lowerText.contains(word.toLowerCase())) {
        return word;
      }
    }
    return null;
  }

  Future<void> _publish() async {
    if (!_canPublish || _previewImagePath == null) return;

    final caption = _descriptionController.text.trim();

    // 🚫 VALIDAR PALABRAS PROHIBIDAS
    if (_containsBannedWords(caption)) {
      final bannedWord = _getFirstBannedWord(caption);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '🚫 Contenido no permitido: La palabra "$bannedWord" está prohibida',
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          backgroundColor: Colors.red[700],
          duration: const Duration(seconds: 4),
        ),
      );
      return; // NO permitir publicar
    }

    try {
      setState(() {
        _publishing = true;
      });
      final user = widget.currentUser;

      await PostService.instance.createPost(
        imageFile: File(_previewImagePath!),
        caption: caption,
        filter: _selectedFilter,
        audioFile: _selectedAudio,
        username: user?.username,
        userPhotoUrl: user?.photoUrl,
      );

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Publicación creada correctamente")),
      );

      _descriptionController.clear();
      setState(() {
        _originalImagePath = null;
        _previewImagePath = null;
        _selectedFilter = null;
        _selectedAudio = null;
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
                    title: const Text('Modo previsualización en vivo (filtros locales)'),
                    subtitle: Text(
                      _originalImagePath != null
                          ? 'Deshabilitado: Ya tienes una foto capturada. Usa filtros CUDA abajo.'
                          : 'Activa para ver filtros en tiempo real con la cámara (filtros locales en Dart).',
                    ),
                    value: _previewEnabled,
                    // Deshabilitar si ya hay una foto capturada
                    onChanged: _originalImagePath != null
                        ? null
                        : (value) {
                            setState(() {
                              _previewEnabled = value;
                              if (value) {
                                // Al activar preview, limpiar cualquier foto previa
                                _originalImagePath = null;
                                _previewImagePath = null;
                                _selectedFilter = null;
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

                  // FILTROS - Dos modos diferentes
                  if (_previewEnabled) ...[
                    // MODO PREVIEW: Filtros locales en Dart (tiempo real)
                    Text(
                      'Filtros en vivo (local - Dart)',
                      style: theme.textTheme.titleSmall!
                          .copyWith(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Estos filtros se aplican en tiempo real con la cámara',
                      style: theme.textTheme.bodySmall!.copyWith(
                        color: Colors.grey[600],
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: _filterOptions.map((filter) {
                        final code = filter['code']!;
                        final selected = _selectedFilter == code;
                        return ChoiceChip(
                          label: Text(filter['label']!),
                          selected: selected,
                          onSelected: (bool value) {
                            setState(() {
                              _selectedFilter = value ? code : null;
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
                  ] else ...[
                    // MODO FOTO CAPTURADA: Filtros CUDA del backend
                    Row(
                      children: [
                        Text(
                          'Filtros CUDA (API Gateway → Backend)',
                          style: theme.textTheme.titleSmall!
                              .copyWith(fontWeight: FontWeight.w600),
                        ),
                        if (_applyingFilter) ...[
                          const SizedBox(width: 8),
                          const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                          const SizedBox(width: 4),
                          Text(
                            'Aplicando...',
                            style: theme.textTheme.bodySmall,
                          ),
                        ],
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _originalImagePath == null
                          ? 'Toma una foto para aplicar filtros GPU del backend'
                          : 'Filtros procesados en el servidor con CUDA',
                      style: theme.textTheme.bodySmall!.copyWith(
                        color: Colors.grey[600],
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 8),
                    if (_originalImagePath == null)
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.grey[200],
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Row(
                          children: [
                            Icon(Icons.info_outline, size: 20, color: Colors.grey[600]),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                'Toma una foto o selecciona de galería para aplicar filtros CUDA',
                                style: TextStyle(
                                  fontSize: 13,
                                  color: Colors.grey[700],
                                ),
                              ),
                            ),
                          ],
                        ),
                      )
                    else
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          // Botón para remover filtro
                          if (_selectedFilter != null)
                            FilterChip(
                              label: const Text('Sin filtro'),
                              selected: false,
                              onSelected: (_applyingFilter) ? null : (_) => _removeFilter(),
                              avatar: const Icon(Icons.close, size: 18),
                            ),
                          // Filtros CUDA disponibles
                          ..._filterOptions.map((filter) {
                            final code = filter['code']!;
                            final selected = _selectedFilter == code;
                            return ChoiceChip(
                              label: Text(filter['label']!),
                              selected: selected,
                              onSelected: _applyingFilter
                                  ? null
                                  : (bool value) {
                                      if (value) {
                                        _applyFilter(code);
                                      } else {
                                        _removeFilter();
                                      }
                                    },
                              selectedColor: theme.colorScheme.primary,
                              labelStyle: TextStyle(
                                color: selected
                                    ? theme.colorScheme.onPrimary
                                    : theme.colorScheme.onSurface,
                              ),
                            );
                          }).toList(),
                        ],
                      ),
                  ],

                  const SizedBox(height: 20),

                  // SELECTOR DE MÚSICA
                  Text(
                    'Música del post (opcional)',
                    style: theme.textTheme.titleSmall!
                        .copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Selecciona una canción para tu publicación',
                    style: theme.textTheme.bodySmall!.copyWith(
                      color: Colors.grey[600],
                      fontSize: 12,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Container(
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey[300]!),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      children: [
                        ListTile(
                          leading: Icon(
                            _selectedAudio == null ? Icons.music_off : Icons.music_note,
                            color: _selectedAudio == null ? Colors.grey : theme.colorScheme.primary,
                          ),
                          title: Text(
                            _selectedAudio == null
                                ? 'Sin música'
                                : _audioOptions.firstWhere((a) => a['file'] == _selectedAudio)['label']!,
                          ),
                          trailing: _selectedAudio == null
                              ? const Icon(Icons.arrow_drop_down)
                              : Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    IconButton(
                                      icon: const Icon(Icons.play_arrow, size: 20),
                                      onPressed: () async {
                                        try {
                                          await _audioPlayer.play(AssetSource('audios/$_selectedAudio'));
                                        } catch (e) {
                                          debugPrint('Error reproduciendo preview: $e');
                                        }
                                      },
                                      tooltip: 'Previsualizar',
                                    ),
                                    IconButton(
                                      icon: const Icon(Icons.close, size: 20),
                                      onPressed: () {
                                        setState(() {
                                          _selectedAudio = null;
                                        });
                                      },
                                      tooltip: 'Quitar música',
                                    ),
                                  ],
                                ),
                          onTap: () {
                            showModalBottomSheet(
                              context: context,
                              builder: (context) => Container(
                                padding: const EdgeInsets.symmetric(vertical: 16),
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Padding(
                                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                                      child: Text(
                                        'Selecciona una canción',
                                        style: theme.textTheme.titleMedium!.copyWith(
                                          fontWeight: FontWeight.w600,
                                        ),
                                      ),
                                    ),
                                    const Divider(),
                                    ListTile(
                                      leading: const Icon(Icons.music_off),
                                      title: const Text('Sin música'),
                                      selected: _selectedAudio == null,
                                      onTap: () {
                                        setState(() {
                                          _selectedAudio = null;
                                        });
                                        Navigator.pop(context);
                                      },
                                    ),
                                    ..._audioOptions.map((audio) {
                                      final file = audio['file']!;
                                      final label = audio['label']!;
                                      final selected = _selectedAudio == file;
                                      return ListTile(
                                        leading: Icon(
                                          Icons.music_note,
                                          color: selected ? theme.colorScheme.primary : null,
                                        ),
                                        title: Text(label),
                                        selected: selected,
                                        trailing: IconButton(
                                          icon: const Icon(Icons.play_arrow, size: 20),
                                          onPressed: () async {
                                            try {
                                              await _audioPlayer.play(AssetSource('audios/$file'));
                                            } catch (e) {
                                              debugPrint('Error reproduciendo: $e');
                                            }
                                          },
                                        ),
                                        onTap: () {
                                          setState(() {
                                            _selectedAudio = file;
                                          });
                                          Navigator.pop(context);
                                        },
                                      );
                                    }).toList(),
                                  ],
                                ),
                              ),
                            );
                          },
                        ),
                      ],
                    ),
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