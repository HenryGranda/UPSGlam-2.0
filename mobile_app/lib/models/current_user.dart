class CurrentUser {
  final String id;
  final String email;
  final String username;
  final String fullName;
  final String? bio;
  final String? photoUrl;
  final int followersCount;
  final int followingCount;

  const CurrentUser({
    required this.id,
    required this.email,
    required this.username,
    required this.fullName,
    this.bio,
    this.photoUrl,
    this.followersCount = 0,
    this.followingCount = 0,
  });

  /// 🔹 Desde Firestore / Backend
  factory CurrentUser.fromJson(Map<String, dynamic> json) {
    return CurrentUser(
      id: json['id'] as String,
      email: json['email'] as String,
      username: json['username'] as String,
      fullName: json['fullName'] as String,
      bio: json['bio'] as String?,
      photoUrl: json['photoUrl'] as String?,
      followersCount: (json['followersCount'] ?? 0) as int,
      followingCount: (json['followingCount'] ?? 0) as int,
    );
  }

  /// Para updates parciales (FOLLOW / UNFOLLOW)
  CurrentUser copyWith({
    String? id,
    String? email,
    String? username,
    String? fullName,
    String? bio,
    String? photoUrl,
    int? followersCount,
    int? followingCount,
  }) {
    return CurrentUser(
      id: id ?? this.id,
      email: email ?? this.email,
      username: username ?? this.username,
      fullName: fullName ?? this.fullName,
      bio: bio ?? this.bio,
      photoUrl: photoUrl ?? this.photoUrl,
      followersCount: followersCount ?? this.followersCount,
      followingCount: followingCount ?? this.followingCount,
    );
  }
}
