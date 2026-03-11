import 'package:flutter/services.dart' show rootBundle;

class FileUtil {
  static Future<List<String>> loadLabels(String labelsFileName) async {
    // Load the labels file from assets
    final labelFile = await rootBundle.loadString('assets/$labelsFileName');

    // Split the file contents into a list by lines and return
    final labels = labelFile.split('\n').map((line) => line.trim()).toList();

    return labels;
  }
}
