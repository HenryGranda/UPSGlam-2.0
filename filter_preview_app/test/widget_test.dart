import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:filter_preview_app/main.dart';

void main() {
  testWidgets('App launches', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const FilterPreviewApp());

    // Verify that the camera screen loads
    expect(find.text('Filter Preview'), findsOneWidget);
  });
}
