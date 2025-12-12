import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../../services/notification_service.dart';

class NotificationsView extends StatefulWidget {
  const NotificationsView({super.key});

  @override
  State<NotificationsView> createState() => _NotificationsViewState();
}

class _NotificationsViewState extends State<NotificationsView> {
  bool _loading = true;
  String? _error;
  List<AppNotification> _items = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final list = await NotificationService.instance.fetchNotifications();
      setState(() {
        _items = list;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  Future<void> _onTap(AppNotification n) async {
    await NotificationService.instance.markAsRead(n.id);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(_titleFor(n))),
    );
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: Colors.white,
            border: Border(bottom: BorderSide(color: colorScheme.outlineVariant)),
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
          child: RefreshIndicator(
            onRefresh: _load,
            child: _buildBody(colorScheme),
          ),
        ),
      ],
    );
  }

  Widget _buildBody(ColorScheme colorScheme) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return ListView(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
              _error!,
              style: TextStyle(color: colorScheme.error),
            ),
          ),
          TextButton(onPressed: _load, child: const Text('Reintentar')),
        ],
      );
    }
    if (_items.isEmpty) {
      return ListView(
        children: const [
          SizedBox(height: 32),
          Center(child: Text('No tienes notificaciones todavía')),
        ],
      );
    }
    return ListView.separated(
      itemCount: _items.length,
      separatorBuilder: (_, __) => const Divider(height: 0),
      itemBuilder: (context, index) {
        final n = _items[index];
        return ListTile(
          leading: CircleAvatar(
            backgroundColor: _colorForType(n.type, colorScheme),
            child: Icon(
              _iconForType(n.type),
              color: Colors.white,
              size: 18,
            ),
          ),
          title: Text(
            _titleFor(n),
            style: TextStyle(
              fontSize: 13,
              fontWeight: n.read ? FontWeight.w400 : FontWeight.w600,
            ),
          ),
          subtitle: Text(
            _timeFor(n),
            style: const TextStyle(fontSize: 11, color: Colors.grey),
          ),
          onTap: () => _onTap(n),
        );
      },
    );
  }

  String _titleFor(AppNotification n) {
    final user = n.actorUsername ?? 'Alguien';
    switch (n.type) {
      case 'like':
        return '$user le dio like a tu publicación';
      case 'comment':
        return '$user comentó tu publicación';
      case 'follow':
        return '$user empezó a seguirte';
      default:
        return 'Actividad de $user';
    }
  }

  String _timeFor(AppNotification n) {
    if (n.createdAt == null) return '';
    final now = DateTime.now();
    final diff = now.difference(n.createdAt!);
    if (diff.inMinutes < 1) return 'Justo ahora';
    if (diff.inMinutes < 60) return '${diff.inMinutes} min';
    if (diff.inHours < 24) return '${diff.inHours} h';
    return DateFormat('dd/MM/yyyy HH:mm').format(n.createdAt!);
  }

  static IconData _iconForType(String type) {
    switch (type) {
      case 'like':
        return Icons.favorite;
      case 'comment':
        return Icons.mode_comment_outlined;
      case 'follow':
        return Icons.person_add_alt_1;
      default:
        return Icons.notifications;
    }
  }

  static Color _colorForType(String type, ColorScheme scheme) {
    switch (type) {
      case 'like':
        return scheme.secondary;
      case 'comment':
        return scheme.primary;
      case 'follow':
        return scheme.tertiary;
      default:
        return scheme.primary;
    }
  }
}
