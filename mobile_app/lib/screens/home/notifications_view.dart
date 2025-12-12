// lib/screens/home/notifications_view.dart
import 'package:flutter/material.dart';

class NotificationsView extends StatelessWidget {
  const NotificationsView({super.key});

  @override
  Widget build(BuildContext context) {
    final notifications = <_NotificationItem>[
      _NotificationItem(
        type: NotificationType.like,
        title: '@maria_ups le dio like a tu publicación',
        time: 'Hace 2 min',
      ),
      _NotificationItem(
        type: NotificationType.comment,
        title: '@carlos_dev comentó: "Brutal ese filtro 🤯"',
        time: 'Hace 10 min',
      ),
      _NotificationItem(
        type: NotificationType.follow,
        title: '@ana_design empezó a seguirte',
        time: 'Hace 1 hora',
      ),
    ];

    return Column(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: const BoxDecoration(
            color: Colors.white,
            border: Border(bottom: BorderSide(color: Color(0xFFE5E7EB))),
          ),
          child: const Center(
            child: Text(
              'Notificaciones',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ),
        Expanded(
          child: ListView.separated(
            itemCount: notifications.length,
            separatorBuilder: (_, __) => const Divider(height: 0),
            itemBuilder: (context, index) {
              final n = notifications[index];
              return ListTile(
                leading: CircleAvatar(
                  backgroundColor: _colorForType(n.type),
                  child: Icon(
                    _iconForType(n.type),
                    color: Colors.white,
                    size: 18,
                  ),
                ),
                title: Text(
                  n.title,
                  style: const TextStyle(fontSize: 13),
                ),
                subtitle: Text(
                  n.time,
                  style: const TextStyle(fontSize: 11, color: Colors.grey),
                ),
                onTap: () {
                  // Más adelante: navegar a la publicación o perfil
                },
              );
            },
          ),
        ),
      ],
    );
  }

  static IconData _iconForType(NotificationType type) {
    switch (type) {
      case NotificationType.like:
        return Icons.favorite;
      case NotificationType.comment:
        return Icons.mode_comment_outlined;
      case NotificationType.follow:
        return Icons.person_add_alt_1;
    }
  }

  static Color _colorForType(NotificationType type) {
    switch (type) {
      case NotificationType.like:
        return Colors.pink;
      case NotificationType.comment:
        return Colors.blueAccent;
      case NotificationType.follow:
        return Colors.green;
    }
  }
}

enum NotificationType { like, comment, follow }

class _NotificationItem {
  final NotificationType type;
  final String title;
  final String time;

  _NotificationItem({
    required this.type,
    required this.title,
    required this.time,
  });
}
