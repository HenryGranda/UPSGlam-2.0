import 'package:flutter/material.dart';
import 'screens/auth/login_screen.dart';
import 'screens/home/upsglam_shell.dart';
import 'services/auth_service.dart';
import 'package:firebase_core/firebase_core.dart';


void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
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
          seedColor: const Color(0xFF0B2545), // azul marino
          primary: const Color(0xFF0B2545),
          secondary: const Color(0xFFB23A48), // tomate oscuro
          tertiary: const Color(0xFF1B3A57),
        ),
        scaffoldBackgroundColor: const Color(0xFFF4F6FA),
      ),
      home: const _AuthGate(),
    );
  }
}

/// =============================
/// AUTH GATE
/// =============================
class _AuthGate extends StatefulWidget {
  const _AuthGate();

  @override
  State<_AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<_AuthGate> {
  bool _loggedIn = false;
  bool _checkingSession = true;

  @override
  void initState() {
    super.initState();
    _checkExistingSession();
  }

  Future<void> _checkExistingSession() async {
    final token = await AuthService.instance.getIdToken();
    if (!mounted) return;
    setState(() {
      _loggedIn = token != null;
      _checkingSession = false;
    });
  }

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
    if (_checkingSession) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

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
