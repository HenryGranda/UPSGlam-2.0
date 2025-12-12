// lib/screens/home/post_detail_page.dart
import 'package:flutter/material.dart';

import '../../models/post_model.dart';
import '../../models/current_user.dart';
import '../../models/comment_model.dart';
import '../../services/post_service.dart';

import 'ui_helpers.dart';

class PostDetailPage extends StatefulWidget {
  final PostModel post;
  final CurrentUser? currentUser;
  final void Function(String postId) onToggleLike;
  final void Function(String postId, CommentModel comment) onAddComment;

  const PostDetailPage({
    super.key,
    required this.post,
    required this.onToggleLike,
    required this.onAddComment,
    this.currentUser,
  });

  @override
  State<PostDetailPage> createState() => _PostDetailPageState();
}

class _PostDetailPageState extends State<PostDetailPage> {
  late PostModel _post;
  final TextEditingController _commentController = TextEditingController();

  Future<void> _loadCommentsFromBackend() async {
    try {
      final comments =
          await PostService.instance.fetchComments(_post.id);

      if (!mounted) return;

      setState(() {
        _post.comments = comments;
      });
    } catch (e) {
      debugPrint('Error cargando comentarios del backend: $e');
    }
  }

  @override
  void initState() {
    super.initState();
    _post = widget.post;
    _loadCommentsFromBackend();
  }

  void _toggleLike() {
    setState(() {
      if (_post.liked) {
        _post.liked = false;
        if (_post.likes > 0) {
          _post.likes -= 1;
        }
      } else {
        _post.liked = true;
        _post.likes += 1;
      }
    });

    try {
      widget.onToggleLike(_post.id);
    } catch (e) {
      // silencioso
    }
  }

  void _openCommentsSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius:
            BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) {
        // StatefulBuilder permite hacer setState SOLO dentro del modal
        return StatefulBuilder(
          builder: (context, modalSetState) {
            // EDITAR comentario
            Future<void> editCommentAt(int index) async {
              final original = _post.comments[index];
              final controller = TextEditingController(text: original.text);

              final edited = await showDialog<String>(
                context: context,
                builder: (ctx) => AlertDialog(
                  title: const Text('Editar comentario'),
                  content: TextField(
                    controller: controller,
                    maxLines: 3,
                    decoration: const InputDecoration(
                      hintText: 'Edita tu comentario…',
                    ),
                  ),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(ctx),
                      child: const Text('Cancelar'),
                    ),
                    TextButton(
                      onPressed: () =>
                          Navigator.pop(ctx, controller.text.trim()),
                      child: const Text('Guardar'),
                    ),
                  ],
                ),
              );

              if (edited == null ||
                  edited.isEmpty ||
                  edited == original.text) return;

              try {
                // 1) actualizar en backend
                final updated = await PostService.instance.updateComment(
                  postId: _post.id,
                  commentId: original.id,
                  newText: edited,
                );

                // 2) actualizar en memoria
                setState(() {
                  _post.comments[index] = updated;
                });
                modalSetState(() {});
              } catch (e) {
                debugPrint('Error editando comentario: $e');
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                      content: Text('No se pudo editar el comentario')),
                );
              }
            }

            // ELIMINAR comentario
            Future<void> deleteCommentAt(int index) async {
              final comment = _post.comments[index];

              final confirm = await showDialog<bool>(
                context: context,
                builder: (ctx) => AlertDialog(
                  title: const Text('Eliminar comentario'),
                  content: const Text(
                      '¿Seguro que quieres eliminar este comentario?'),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(ctx, false),
                      child: const Text('Cancelar'),
                    ),
                    TextButton(
                      onPressed: () => Navigator.pop(ctx, true),
                      child: const Text(
                        'Eliminar',
                        style: TextStyle(color: Colors.red),
                      ),
                    ),
                  ],
                ),
              );

              if (confirm != true) return;

              try {
                await PostService.instance.deleteComment(
                  postId: _post.id,
                  commentId: comment.id,
                );

                setState(() {
                  _post.comments.removeAt(index);
                });
                modalSetState(() {});
              } catch (e) {
                debugPrint('Error eliminando comentario: $e');
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('No se pudo eliminar el comentario'),
                  ),
                );
              }
            }

            Future<void> addCommentFromSheet() async {
              final text = _commentController.text.trim();
              if (text.isEmpty) return;

              try {
                // 1) Crear en backend y obtener el CommentModel "real"
                final created = await PostService.instance.addComment(
                  postId: _post.id,
                  text: text,
                );

                // 2) Actualizar el detalle local
                setState(() {
                  _post.comments.add(created);
                });

                // 3) Avisar al shell para que actualice el feed
                widget.onAddComment(_post.id, created);

                // 4) Refrescar el bottom sheet
                modalSetState(() {});

                // 5) Limpiar caja de texto
                _commentController.clear();
              } catch (e) {
                debugPrint('Error al crear comentario: $e');
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                      content: Text('Error al publicar comentario')),
                );
              }
            }

            return Padding(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).viewInsets.bottom,
                top: 8,
              ),
              child: SizedBox(
                height: MediaQuery.of(context).size.height * 0.75,
                child: Column(
                  children: [
                    // Handle
                    Container(
                      width: 40,
                      height: 4,
                      margin: const EdgeInsets.only(top: 8, bottom: 12),
                      decoration: BoxDecoration(
                        color: Colors.grey.shade300,
                        borderRadius: BorderRadius.circular(999),
                      ),
                    ),
                    const Text(
                      'Comentarios',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        fontSize: 16,
                      ),
                    ),
                    const SizedBox(height: 8),

                    // Lista de comentarios
                    Expanded(
                      child: _post.comments.isEmpty
                          ? const Center(
                              child: Text(
                                'No hay comentarios aún',
                                style: TextStyle(
                                    fontSize: 13, color: Colors.grey),
                              ),
                            )
                          : ListView.builder(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 16, vertical: 8),
                              itemCount: _post.comments.length,
                              itemBuilder: (context, index) {
                                final c = _post.comments[index];
                                return Padding(
                                  padding: const EdgeInsets.symmetric(
                                      vertical: 6),
                                  child: Row(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      CircleAvatar(
                                        radius: 16,
                                        backgroundImage:
                                            buildImageProvider(c.avatar),
                                      ),
                                      const SizedBox(width: 8),
                                      Expanded(
                                        child: Column(
                                          crossAxisAlignment:
                                              CrossAxisAlignment.start,
                                          children: [
                                            Row(
                                              children: [
                                                Text(
                                                  c.username,
                                                  style: const TextStyle(
                                                    fontWeight:
                                                        FontWeight.w600,
                                                    fontSize: 13,
                                                  ),
                                                ),
                                                const SizedBox(width: 6),
                                                Text(
                                                  c.timestamp,
                                                  style: const TextStyle(
                                                    fontSize: 11,
                                                    color: Colors.grey,
                                                  ),
                                                ),
                                              ],
                                            ),
                                            const SizedBox(height: 2),
                                            Text(
                                              c.text,
                                              style: const TextStyle(
                                                fontSize: 13,
                                              ),
                                            ),
                                          ],
                                        ),
                                      ),
                                      PopupMenuButton<String>(
                                        onSelected: (value) {
                                          if (value == 'edit') {
                                            editCommentAt(index);
                                          } else if (value == 'delete') {
                                            deleteCommentAt(index);
                                          }
                                        },
                                        itemBuilder: (context) => const [
                                          PopupMenuItem(
                                            value: 'edit',
                                            child: Text('Editar'),
                                          ),
                                          PopupMenuItem(
                                            value: 'delete',
                                            child: Text('Eliminar'),
                                          ),
                                        ],
                                      ),
                                    ],
                                  ),
                                );
                              },
                            ),
                    ),

                    // Caja para escribir comentario
                    Padding(
                      padding:
                          const EdgeInsets.fromLTRB(16, 8, 16, 16),
                      child: Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: _commentController,
                              decoration: InputDecoration(
                                hintText: 'Añadir un comentario…',
                                border: OutlineInputBorder(
                                  borderRadius:
                                      BorderRadius.circular(20),
                                ),
                                contentPadding:
                                    const EdgeInsets.symmetric(
                                  horizontal: 12,
                                  vertical: 8,
                                ),
                              ),
                              onSubmitted: (_) =>
                                  addCommentFromSheet(),
                            ),
                          ),
                          const SizedBox(width: 8),
                          IconButton(
                            onPressed: addCommentFromSheet,
                            icon: const Icon(Icons.send),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final displayUsername = formatUsername(_post.username);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Publicación'),
        centerTitle: true,
      ),
      body: Column(
        children: [
          // Info usuario
          Padding(
            padding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            child: Row(
              children: [
                CircleAvatar(
                  radius: 22,
                  backgroundImage: buildImageProvider(_post.avatar),
                ),
                const SizedBox(width: 8),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      displayUsername,
                      style: const TextStyle(
                        fontWeight: FontWeight.w600,
                        fontSize: 15,
                      ),
                    ),
                    Text(
                      _post.timestamp,
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.grey,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Imagen principal
          AspectRatio(
            aspectRatio: 1,
            child: Image(
              image: buildImageProvider(_post.imageUrl),
              fit: BoxFit.cover,
            ),
          ),

          // Info abajo
          Padding(
            padding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Botones like / comentarios / enviar
                Row(
                  children: [
                    InkWell(
                      onTap: _toggleLike,
                      child: Row(
                        children: [
                          Icon(
                            _post.liked
                                ? Icons.favorite
                                : Icons.favorite_border_outlined,
                            color: _post.liked
                                ? Colors.pink
                                : Colors.black87,
                            size: 28,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            '${_post.likes}',
                            style: const TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 16),
                    InkWell(
                      onTap: _openCommentsSheet,
                      child: Row(
                        children: [
                          const Icon(Icons.mode_comment_outlined,
                              size: 26),
                          const SizedBox(width: 4),
                          Text(
                            '${_post.comments.length}',
                            style: const TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 16),
                    const Icon(Icons.send_outlined, size: 26),
                  ],
                ),

                const SizedBox(height: 8),

                // Nº me gusta
                Text(
                  '${_post.likes} me gusta',
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),

                const SizedBox(height: 6),

                // Descripción
                RichText(
                  text: TextSpan(
                    style: const TextStyle(
                      fontSize: 13,
                      color: Colors.black87,
                    ),
                    children: [
                      TextSpan(
                        text: displayUsername,
                        style:
                            const TextStyle(fontWeight: FontWeight.w600),
                      ),
                      const TextSpan(text: ' '),
                      TextSpan(
                        text: _post.caption,
                        style:
                            const TextStyle(color: Colors.black87),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 6),

                // Filtro aplicado (solo nombre, sin efecto visual)
                if (_post.filter != null)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.purple.shade50,
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Text(
                      'Filtro aplicado: ${_post.filter}',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: Colors.purple.shade700,
                      ),
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
