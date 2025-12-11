class CurrentUser {
  final String id;
  final String email;
  final String username;
  final String fullName;
  final String? bio;
  final String? photoUrl;

  CurrentUser({
    required this.id,
    required this.email,
    required this.username,
    required this.fullName,
    this.bio,
    this.photoUrl,
  });

  factory CurrentUser.fromJson(Map<String, dynamic> json) {
    return CurrentUser(
      id: json['id'] ?? '',
      email: json['email'] ?? '',
      username: json['username'] ?? '',
      fullName: json['fullName'] ?? '',
      bio: json['bio'],
      photoUrl: json['photoUrl'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      'username': username,
      'fullName': fullName,
      'bio': bio,
      'photoUrl': photoUrl,
    };
  }

  CurrentUser copyWith({
    String? username,
    String? fullName,
    String? bio,
    String? photoUrl,
  }) {
    return CurrentUser(
      id: id,
      email: email,
      username: username ?? this.username,
      fullName: fullName ?? this.fullName,
      bio: bio ?? this.bio,
      photoUrl: photoUrl ?? this.photoUrl,
    );
  }
}
