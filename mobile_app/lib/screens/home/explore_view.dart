// lib/screens/home/explore_view.dart
import 'package:flutter/material.dart';

import '../../models/post_model.dart';
import 'post_detail_page.dart';
import 'ui_helpers.dart';

class ExploreView extends StatefulWidget {
  final List<PostModel> posts;

  const ExploreView({super.key, required this.posts});

  @override
  State<ExploreView> createState() => _ExploreViewState();
}

class _ExploreViewState extends State<ExploreView> {
  String _search = '';

  @override
  Widget build(BuildContext context) {
    final filtered = widget.posts.where((p) {
      if (_search.trim().isEmpty) return true;
      final q = _search.toLowerCase();
      return formatUsername(p.username).toLowerCase().contains(q) ||
          p.caption.toLowerCase().contains(q);
    }).toList();

    return Column(
      children: [
        // Header + buscador
        Container(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
          decoration: const BoxDecoration(
            color: Colors.white,
            border: Border(
              bottom: BorderSide(color: Color(0xFFE5E7EB)),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Explorar',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 8),
              TextField(
                decoration: InputDecoration(
                  prefixIcon: const Icon(Icons.search),
                  hintText: 'Buscar usuarios o publicaciones…',
                  isDense: true,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                ),
                onChanged: (value) {
                  setState(() => _search = value);
                },
              ),
            ],
          ),
        ),

        // Grid de imágenes
        Expanded(
          child: filtered.isEmpty
              ? const Center(
                  child: Text(
                    'No se encontraron resultados',
                    style: TextStyle(color: Colors.grey),
                  ),
                )
              : GridView.builder(
                  padding: const EdgeInsets.all(1),
                  gridDelegate:
                      const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 3,
                    mainAxisSpacing: 1,
                    crossAxisSpacing: 1,
                  ),
                  itemCount: filtered.length,
                  itemBuilder: (context, index) {
                    final post = filtered[index];
                    return GestureDetector(
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => PostDetailPage(
                              post: post,
                              currentUser: null,
                              onToggleLike: (id) {},
                              onAddComment: (id, c) {},
                            ),
                          ),
                        );
                      },
                      child: AspectRatio(
                        aspectRatio: 1,
                        child: Image(
                          image: buildImageProvider(post.imageUrl),
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) => Container(
                            color: Colors.grey.shade300,
                            child: const Icon(Icons.image, size: 30),
                          ),
                        ),
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }
}
