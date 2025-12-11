import 'package:flutter/material.dart';
import '../../services/auth_service.dart';
import '../../models/current_user.dart';

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

  @override
  void initState() {
    super.initState();
    _usernameCtrl = TextEditingController(text: widget.currentUser.username);
    _fullNameCtrl = TextEditingController(text: widget.currentUser.fullName);
    _bioCtrl = TextEditingController(text: widget.currentUser.bio ?? '');
  }

  Future<void> _save() async {
    final fullName = _fullNameCtrl.text.trim();
    final username = _usernameCtrl.text.trim().replaceAll('@', '');
    final bio = _bioCtrl.text.trim();

    final original = widget.currentUser!;

    final usernameToSend =
        username != original.username ? username : null;

    try {
      await AuthService.instance.updateProfile(
        fullName: fullName,
        username: usernameToSend, 
        bio: bio,
      );

      Navigator.pop(context, true);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error al actualizar perfil: $e")),
      );
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
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
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
