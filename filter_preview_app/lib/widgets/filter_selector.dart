import 'package:flutter/material.dart';

class FilterSelector extends StatelessWidget {
  final String selectedFilter;
  final Function(String) onFilterSelected;

  const FilterSelector({
    super.key,
    required this.selectedFilter,
    required this.onFilterSelected,
  });

  @override
  Widget build(BuildContext context) {
    final filters = [
      {'id': 'gaussian', 'name': 'Gauss', 'icon': Icons.blur_on},
      {'id': 'box_blur', 'name': 'Blox Blur', 'icon': Icons.blur_circular},
      {'id': 'prewitt', 'name': 'Prewitt', 'icon': Icons.grid_on},
      {'id': 'laplacian', 'name': 'Laplace', 'icon': Icons.highlight},
      {'id': 'ups_logo', 'name': 'UPS Logo', 'icon': Icons.shield},
      {'id': 'boomerang', 'name': 'Boomerang', 'icon': Icons.auto_awesome},
      {'id': 'caras', 'name': 'Caras', 'icon': Icons.face_retouching_natural},
    ];

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Wrap(
        alignment: WrapAlignment.center,
        spacing: 16,
        runSpacing: 16,
        children: filters.map((filter) {
          final isSelected = selectedFilter == filter['id'];

          return GestureDetector(
            onTap: () => onFilterSelected(filter['id'] as String),
            child: Container(
              width: 80,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 60,
                    height: 60,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: isSelected
                          ? const Color(0xFFF2A900)
                          : Colors.white.withOpacity(0.2),
                      border: Border.all(
                        color: Colors.white,
                        width: isSelected ? 3 : 1,
                      ),
                    ),
                    child: Icon(
                      filter['icon'] as IconData,
                      color: Colors.white,
                      size: 28,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    filter['name'] as String,
                    style: TextStyle(
                      color: isSelected ? const Color(0xFFF2A900) : Colors.white,
                      fontSize: 12,
                      fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}
