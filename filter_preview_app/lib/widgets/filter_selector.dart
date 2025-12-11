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
      {'id': 'gaussian', 'name': 'Gaussian', 'icon': Icons.blur_on},
      {'id': 'box_blur', 'name': 'Box Blur', 'icon': Icons.blur_circular},
      {'id': 'prewitt', 'name': 'Prewitt', 'icon': Icons.grid_on},
      {'id': 'laplacian', 'name': 'Laplacian', 'icon': Icons.highlight},
      {'id': 'ups_logo', 'name': 'UPS Logo', 'icon': Icons.school},
      {'id': 'ups_color', 'name': 'UPS Color', 'icon': Icons.palette},
      {'id': 'boomerang', 'name': 'Boomerang', 'icon': Icons.adjust},
    ];

    return SizedBox(
      height: 100,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        itemCount: filters.length,
        itemBuilder: (context, index) {
          final filter = filters[index];
          final isSelected = selectedFilter == filter['id'];

          return GestureDetector(
            onTap: () => onFilterSelected(filter['id'] as String),
            child: Container(
              width: 80,
              margin: const EdgeInsets.symmetric(horizontal: 8),
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
        },
      ),
    );
  }
}
