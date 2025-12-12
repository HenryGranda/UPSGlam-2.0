// lib/models/avatars.dart

const kAvatarNames = <String>[
  'avatar1.jpeg',
  'avatar2.jpeg',
  'avatar3.jpeg',
  'avatar4.jpeg',
  'avatar5.jpeg',
  'avatar6.jpeg',
  'avatar7.jpeg',
  'avatar8.jpeg',
];

const String kDefaultAvatarAsset = 'assets/avatars/avatar1.jpeg';

String avatarAssetFromName(String? name) {
  if (name == null || name.isEmpty) {
    return kDefaultAvatarAsset;
  }

  // Si ya viene con ruta (assets/... o http...), la dejamos tal cual
  if (name.contains('/')) {
    return name;
  }

  // Caso normal: solo nombre de archivo tipo "avatar3.jpeg"
  return 'assets/avatars/$name';
}
