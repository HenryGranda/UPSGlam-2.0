import 'package:flutter/material.dart';
import '../../services/api_config.dart';

class IpConfigScreen extends StatefulWidget {
  const IpConfigScreen({super.key});

  @override
  State<IpConfigScreen> createState() => _IpConfigScreenState();
}

class _IpConfigScreenState extends State<IpConfigScreen> {
  final TextEditingController _ipController = TextEditingController();
  bool _isSaving = false;

  @override
  void initState() {
    super.initState();
    _loadCurrentIp();
  }

  Future<void> _loadCurrentIp() async {
    final current = await ApiConfig.getBaseUrl();
    if (current != null) {
      _ipController.text = current;
    }
  }

  Future<void> _save() async {
    var text = _ipController.text.trim();
    if (text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Ingresa la IP o URL del backend')),
      );
      return;
    }

    // Si el usuario pone solo 192.168.x.x:8080, le agregamos http://
    if (!text.startsWith('http://') && !text.startsWith('https://')) {
      text = 'http://$text';
    }

    // Asegurar que tenga /api al final (según la doc del gateway)
    if (!text.endsWith('/api')) {
      // por si acaso ya puso una barra al final
      text = text.replaceAll(RegExp(r'/+$'), '');
      text = '$text/api';
    }

    setState(() => _isSaving = true);
    await ApiConfig.setBaseUrl(text);
    setState(() => _isSaving = false);

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Backend guardado: $text')),
    );

    Navigator.pop(context); // volvemos al login
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Configurar IP del Backend'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Ingresa la IP o URL del API Gateway de UPSGlam 2.0.\n'
              'Ejemplo: http://192.168.1.10:8080/api',
              style: TextStyle(fontSize: 14),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _ipController,
              decoration: const InputDecoration(
                labelText: 'IP o URL del backend Java',
                hintText: 'http://192.168.1.10:8080/api',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Debe apuntar al API Gateway (puerto 8080) e incluir el sufijo /api.',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            const Spacer(),
            FilledButton(
              onPressed: _isSaving ? null : _save,
              style: FilledButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
              child: _isSaving
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Guardar'),
            ),
            const SizedBox(height: 12),
            OutlinedButton(
              onPressed: _isSaving ? null : () => Navigator.pop(context),
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                foregroundColor: colorScheme.primary,
              ),
              child: const Text('Cancelar / Volver'),
            ),
          ],
        ),
      ),
    );
  }
}
