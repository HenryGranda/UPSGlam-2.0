import 'package:flutter/material.dart';
import '../../services/api_config.dart';
import '../../services/auth_service.dart';
import '../ip_config/ip_config_screen.dart';
import '../auth/register_screen.dart';


class LoginScreen extends StatefulWidget {
  final VoidCallback onLoginSuccess;

  const LoginScreen({super.key, required this.onLoginSuccess});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _identifierController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  String? _backendIp;
  bool _loadingIp = true;
  bool _loggingIn = false;

  @override
  void initState() {
    super.initState();
    _loadBackendIp();
  }

  Future<void> _loadBackendIp() async {
    final ip = await ApiConfig.getBaseUrl();
    setState(() {
      _backendIp = ip;
      _loadingIp = false;
    });
  }

  Future<void> _openSettings() async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const IpConfigScreen()),
    );
    await _loadBackendIp();
  }

  bool get _backendConfigured => (_backendIp ?? '').isNotEmpty;

  Future<void> _loginWithEmailPassword() async {
    if (!_backendConfigured) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Primero configura la IP del backend')),
      );
      return;
    }

    final identifier = _identifierController.text.trim();
    final password = _passwordController.text.trim();

    if (identifier.isEmpty || password.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Ingresa usuario/correo y contraseña')),
      );
      return;
    }

    setState(() => _loggingIn = true);
    try {
      await AuthService.instance
          .loginWithEmailPassword(identifier, password);

      widget.onLoginSuccess();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error al iniciar sesión: $e')),
      );
    } finally {
      setState(() => _loggingIn = false);
    }
  }

  Future<void> _loginWithGoogle() async {
    if (!_backendConfigured) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Primero configura la IP del backend')),
      );
      return;
    }

    setState(() => _loggingIn = true);
    try {
      //await AuthService.instance.loginWithGoogle();
      widget.onLoginSuccess();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error con Google: $e')),
      );
    } finally {
      setState(() => _loggingIn = false);
    }
  }

  Future<void> _loginWithFacebook() async {
    if (!_backendConfigured) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Primero configura la IP del backend')),
      );
      return;
    }

    setState(() => _loggingIn = true);
    try {
      //await AuthService.instance.loginWithFacebook();
      widget.onLoginSuccess();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error con Facebook: $e')),
      );
    } finally {
      setState(() => _loggingIn = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            children: [
              Align(
                alignment: Alignment.topRight,
                child: IconButton(
                  onPressed: _openSettings,
                  icon: const Icon(Icons.settings_outlined),
                  color: colorScheme.primary,
                ),
              ),
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    children: [
                      const SizedBox(height: 16),
                      Container(
                        height: 120,
                        width: 120,
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(24),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.06),
                              blurRadius: 18,
                              offset: const Offset(0, 8),
                            ),
                          ],
                        ),
                        alignment: Alignment.center,
                        child: Image.asset(
                          'assets/images/logoups.png',
                          fit: BoxFit.contain,
                          errorBuilder: (_, __, ___) =>
                              const FlutterLogo(size: 80),
                        ),
                      ),
                      const SizedBox(height: 20),
                      Text(
                        'UPSGlam 2.0',
                        style: TextStyle(
                          fontSize: 32,
                          fontWeight: FontWeight.bold,
                          color: colorScheme.primary,
                        ),
                      ),
                      const SizedBox(height: 4),
                      const Text(
                        'Social GPU Image Platform for UPS students',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.grey,
                        ),
                      ),
                      const SizedBox(height: 32),

                      // IDENTIFIER
                      Align(
                        alignment: Alignment.centerLeft,
                        child: Text(
                          'Usuario o correo',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: colorScheme.onSurface,
                          ),
                        ),
                      ),
                      const SizedBox(height: 6),
                      TextField(
                        controller: _identifierController,
                        decoration: InputDecoration(
                          hintText: 'usuario_ups o correo@ups.edu.ec',
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 14,
                            vertical: 10,
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),

                      // PASSWORD
                      Align(
                        alignment: Alignment.centerLeft,
                        child: Text(
                          'Contraseña',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: colorScheme.onSurface,
                          ),
                        ),
                      ),
                      const SizedBox(height: 6),
                      TextField(
                        controller: _passwordController,
                        obscureText: true,
                        decoration: InputDecoration(
                          hintText: '••••••••',
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 14,
                            vertical: 10,
                          ),
                        ),
                      ),
                      const SizedBox(height: 24),

                      SizedBox(
                        width: double.infinity,
                        child: FilledButton(
                          onPressed:
                              _loggingIn ? null : _loginWithEmailPassword,
                          style: FilledButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            textStyle: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          child: _loggingIn
                              ? const SizedBox(
                                  height: 22,
                                  width: 22,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                )
                              : const Text('Iniciar sesión'),
                        ),
                      ),
                      const SizedBox(height: 12),

                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (_) => const RegisterScreen()),
                          );
                        },
                        child: const Text(
                          '¿No tienes cuenta? Registrarte',
                          style: TextStyle(fontSize: 13),
                        ),
                      ),
                      const SizedBox(height: 16),

                      // separador + Google/Facebook = igual que tenías
                      Row(
                        children: const [
                          Expanded(
                            child: Divider(
                              thickness: 0.8,
                              color: Color(0xFFE0E0E0),
                            ),
                          ),
                          Padding(
                            padding: EdgeInsets.symmetric(horizontal: 8.0),
                            child: Text(
                              'O continúa con',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.grey,
                              ),
                            ),
                          ),
                          Expanded(
                            child: Divider(
                              thickness: 0.8,
                              color: Color(0xFFE0E0E0),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      Row(
                        children: [
                          Expanded(
                            child: OutlinedButton.icon(
                              onPressed:
                                  _loggingIn ? null : _loginWithGoogle,
                              style: OutlinedButton.styleFrom(
                                padding:
                                    const EdgeInsets.symmetric(vertical: 12),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                              ),
                              icon: Image.asset(
                                'assets/icons/google.png',
                                height: 18,
                                errorBuilder: (_, __, ___) =>
                                    const Icon(Icons.g_mobiledata),
                              ),
                              label: const Text(
                                'Google',
                                style: TextStyle(fontSize: 14),
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: OutlinedButton.icon(
                              onPressed:
                                  _loggingIn ? null : _loginWithFacebook,
                              style: OutlinedButton.styleFrom(
                                padding:
                                    const EdgeInsets.symmetric(vertical: 12),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                              ),
                              icon: const Icon(
                                Icons.facebook,
                                color: Colors.blue,
                              ),
                              label: const Text(
                                'Facebook',
                                style: TextStyle(fontSize: 14),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 24),

                      if (_loadingIp)
                        const CircularProgressIndicator(strokeWidth: 2)
                      else
                        Text(
                          _backendIp == null || _backendIp!.isEmpty
                              ? 'IP de backend no configurada'
                              : 'Backend: $_backendIp',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                            color: _backendIp == null || _backendIp!.isEmpty
                                ? Colors.red
                                : Colors.green[700],
                          ),
                        ),
                      const SizedBox(height: 16),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
