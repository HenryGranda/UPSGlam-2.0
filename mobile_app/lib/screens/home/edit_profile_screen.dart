import 'package:flutter/material.dart';
import '../../services/auth_service.dart';
import '../../models/current_user.dart';
import '../../models/avatars.dart'; // 👈 donde están kAvatarNames y avatarAssetFromName

class EditProfileScreen extends StatefulWidget {
  final CurrentUser currentUser;

  const EditProfileScreen({super.key, required this.currentUser});

  @override
  State<EditProfileScreen> createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  late TextEditingController _usernameCtrl;
  late TextEditingController _fullNameCtrl;
  late TextEditingController _bioCtrl;
  bool _saving = false;

  /// nombre del avatar seleccionado: "avatar3.jpeg"
  String? _selectedAvatarName;

  @override
  void initState() {
    super.initState();
    _usernameCtrl = TextEditingController(text: widget.currentUser.username);
    _fullNameCtrl = TextEditingController(text: widget.currentUser.fullName);
    _bioCtrl = TextEditingController(text: widget.currentUser.bio ?? '');

    // Normalizar el photoUrl que viene del backend:
    // puede ser "avatar3.jpeg" o "assets/avatars/avatar3.jpeg" o null
    final rawPhoto = widget.currentUser.photoUrl;
    if (rawPhoto != null && rawPhoto.isNotEmpty) {
      _selectedAvatarName = rawPhoto.split('/').last;
    } else {
      _selectedAvatarName = kAvatarNames.first; // por defecto avatar1
    }
  }

  String _currentAvatarAsset() {
    final name = _selectedAvatarName ?? kAvatarNames.first;
    return avatarAssetFromName(name);
  }

  void _onAvatarTap(String name) {
    setState(() {
      _selectedAvatarName = name;
    });
  }

  Future<void> _save() async {
    if (_saving) return;
    setState(() => _saving = true);

    final original = widget.currentUser;

    final fullName = _fullNameCtrl.text.trim();
    final username = _usernameCtrl.text.trim().replaceAll('@', '');
    final bio = _bioCtrl.text.trim();

    // Solo mando estos si cambiaron
    final fullNameToSend =
        fullName.isNotEmpty && fullName != original.fullName ? fullName : null;

    final usernameToSend =
        username.isNotEmpty && username != original.username ? username : null;

    final bioToSend =
        bio != (original.bio ?? '') ? bio : null;

    final photoToSend = _selectedAvatarName; 

    try {
      await AuthService.instance.updateProfile(
        fullName: fullNameToSend,
        username: usernameToSend,
        bio: bioToSend,
        photoUrl: photoToSend,
      );

      if (!mounted) return;
      Navigator.pop(context, true);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error al actualizar perfil: $e')),
      );
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Editar perfil'),
        actions: [
          TextButton(
            onPressed: _saving ? null : _save,
            child: _saving
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Text('Guardar'),
          )
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // PREVIEW AVATAR
            Center(
              child: Column(
                children: [
                  CircleAvatar(
                    radius: 42,
                    backgroundImage: AssetImage(_currentAvatarAsset()),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Foto de perfil',
                    style: TextStyle(
                      fontSize: 13,
                      color: cs.onSurface.withOpacity(0.6),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // GRID AVATARES
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                'Elige un avatar',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: cs.onSurface.withOpacity(0.8),
                ),
              ),
            ),
            const SizedBox(height: 8),
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: kAvatarNames.length,
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 4,
                mainAxisSpacing: 8,
                crossAxisSpacing: 8,
              ),
              itemBuilder: (_, index) {
                final name = kAvatarNames[index];
                final assetPath = avatarAssetFromName(name);
                final isSelected = _selectedAvatarName == name;

                return GestureDetector(
                  onTap: () => _onAvatarTap(name),
                  child: Column(
                    children: [
                      CircleAvatar(
                        radius: isSelected ? 26 : 24,
                        backgroundImage: AssetImage(assetPath),
                      ),
                      if (isSelected)
                        const Padding(
                          padding: EdgeInsets.only(top: 2),
                          child: Icon(
                            Icons.check_circle,
                            size: 16,
                            color: Colors.purple,
                          ),
                        ),
                    ],
                  ),
                );
              },
            ),

            const SizedBox(height: 16),

            TextField(
              controller: _fullNameCtrl,
              decoration: const InputDecoration(
                labelText: 'Nombre completo',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _usernameCtrl,
              decoration: const InputDecoration(
                labelText: 'Usuario',
                prefixText: '@',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _bioCtrl,
              maxLines: 3,
              decoration: const InputDecoration(
                labelText: 'Biografía',
                hintText: 'Añade una biografía para que te conozcan mejor ✍️',
                border: OutlineInputBorder(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
