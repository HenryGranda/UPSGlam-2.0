import 'package:flutter/material.dart';
import 'screens/auth/login_screen.dart';
import 'screens/home/upsglam_shell.dart';
import 'services/auth_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Aquí luego tu amigo inicializa Firebase.initializeApp()
  runApp(const UPSGlamApp());
}

class UPSGlamApp extends StatelessWidget {
  const UPSGlamApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'UPSGlam 2.0',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF003366), // azul UPS aprox
        ),
        scaffoldBackgroundColor: const Color(0xFFF5F7FB),
      ),
      home: const _AuthGate(),
    );
  }
}

/// =============================
/// AUTH GATE (decide login vs app)
/// =============================
class _AuthGate extends StatefulWidget {
  const _AuthGate();

  @override
  State<_AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<_AuthGate> {
  bool _loggedIn = false;

  void _handleLoginSuccess() {
    setState(() {
      _loggedIn = true;
    });
  }

  Future<void> _handleLogout() async {
    await AuthService.instance.logout();
    setState(() {
      _loggedIn = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_loggedIn) {
      return LoginScreen(
        onLoginSuccess: _handleLoginSuccess,
      );
    }

    return UPSGlamShell(
      onLogout: _handleLogout,
    );
  }
}
