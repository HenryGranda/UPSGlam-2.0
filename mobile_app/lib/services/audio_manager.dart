// lib/services/audio_manager.dart
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';

/// Singleton para gestionar la reproducción de audio
/// Solo permite un audio activo a la vez (como TikTok/Instagram)
class AudioManager {
  static final AudioManager _instance = AudioManager._internal();
  factory AudioManager() => _instance;
  AudioManager._internal();

  AudioPlayer? _currentPlayer;
  String? _currentPostId;

  /// Registra y reproduce un audio para un post específico
  /// Pausa cualquier audio anterior que esté sonando
  Future<void> playAudio({
    required String postId,
    required AudioPlayer player,
    required String audioFile,
  }) async {
    debugPrint('🎵 AudioManager: Solicitud de reproducción para post $postId');

    // Si es el mismo post que ya está sonando, no hacer nada
    if (_currentPostId == postId && _currentPlayer == player) {
      debugPrint('✅ Ya está sonando este post');
      return;
    }

    // Pausar el audio anterior
    if (_currentPlayer != null && _currentPostId != postId) {
      debugPrint('⏸️ Pausando audio del post $_currentPostId');
      await _currentPlayer!.pause();
    }

    // Actualizar referencia al nuevo player
    _currentPlayer = player;
    _currentPostId = postId;

    // Reproducir el nuevo audio
    debugPrint('▶️ Reproduciendo audio del post $postId');
    await player.play(AssetSource('audios/$audioFile'));
  }

  /// Pausa el audio actual si coincide con el postId
  Future<void> pauseAudio(String postId) async {
    if (_currentPostId == postId && _currentPlayer != null) {
      debugPrint('⏸️ Pausando audio del post $postId');
      await _currentPlayer!.pause();
      _currentPlayer = null;
      _currentPostId = null;
    }
  }

  /// Detiene completamente el audio actual
  Future<void> stopCurrent() async {
    if (_currentPlayer != null) {
      debugPrint('⏹️ Deteniendo audio del post $_currentPostId');
      await _currentPlayer!.stop();
      _currentPlayer = null;
      _currentPostId = null;
    }
  }

  /// Verifica si un post específico es el que está sonando
  bool isPlaying(String postId) {
    return _currentPostId == postId;
  }
}
