import 'package:flutter/material.dart';
import 'screens/camera_screen.dart';

void main() {
  runApp(const FilterPreviewApp());
}

class FilterPreviewApp extends StatelessWidget {
  const FilterPreviewApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'UPS Glam Filters',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFF2A900), // UPS Gold
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const CameraScreen(),
    );
  }
}
