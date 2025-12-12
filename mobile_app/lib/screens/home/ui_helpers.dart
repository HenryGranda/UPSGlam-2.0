// lib/screens/home/ui_helpers.dart
import 'dart:io';
import 'package:flutter/material.dart';

import '../../models/avatars.dart';

/// Helper para normalizar @username
String formatUsername(String? raw) {
  if (raw == null || raw.isEmpty) return '@usuario_ups';
  return raw.startsWith('@') ? raw : '@$raw';
}

/// Construye un ImageProvider según el path/url
ImageProvider buildImageProvider(String path) {
  // Si viene vacío → avatar por defecto
  if (path.isEmpty) {
    return const AssetImage('assets/avatars/avatar1.jpeg');
  }

  if (kAvatarNames.contains(path)) {
    return AssetImage(avatarAssetFromName(path));
  }

  if (path.startsWith('assets/')) {
    return AssetImage(path);
  }

  if (path.startsWith('http')) {
    return NetworkImage(path);
  }

  return FileImage(File(path));
}
