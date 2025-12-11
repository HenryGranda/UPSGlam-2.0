import 'dart:typed_data';
import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final Uint8List originalImage;
  final Uint8List processedImage;
  final String filterName;

  const ResultScreen({
    super.key,
    required this.originalImage,
    required this.processedImage,
    required this.filterName,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: Text('Resultado: $filterName'),
      ),
      body: Column(
        children: [
          // Processed Image
          Expanded(
            flex: 3,
            child: Center(
              child: Image.memory(
                processedImage,
                fit: BoxFit.contain,
              ),
            ),
          ),

          // Info
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.black87,
            child: Column(
              children: [
                const Text(
                  'Imagen procesada con PyCUDA',
                  style: TextStyle(
                    color: Color(0xFFF2A900),
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Filtro: $filterName',
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 14,
                  ),
                ),
              ],
            ),
          ),

          // Action Buttons
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.black,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                // Back Button
                ElevatedButton.icon(
                  onPressed: () => Navigator.of(context).pop(),
                  icon: const Icon(Icons.arrow_back),
                  label: const Text('Volver'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.grey[800],
                    foregroundColor: Colors.white,
                  ),
                ),

                // Save Button (placeholder)
                ElevatedButton.icon(
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Función de guardar no implementada'),
                      ),
                    );
                  },
                  icon: const Icon(Icons.save),
                  label: const Text('Guardar'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFF2A900),
                    foregroundColor: Colors.black,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
