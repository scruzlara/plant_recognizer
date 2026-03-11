// Copyright (c) 2022 Kodeco LLC

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// Notwithstanding the foregoing, you may not use, copy, modify, merge, publish,
// distribute, sublicense, create a derivative work, and/or sell copies of the
// Software in any work that is designed, intended, or marketed for pedagogical
// or instructional purposes related to programming, coding,
// application development, or information technology.  Permission for such use,
// copying, modification, merger, publication, distribution, sublicensing,
// creation of derivative works, or sale is expressly withheld.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'classifier_category.dart';
import 'classifier_model.dart';

typedef ClassifierLabels = List<String>;

class Classifier {
  final ClassifierLabels _labels;
  final ClassifierModel _model;

  Classifier._({
    required ClassifierLabels labels,
    required ClassifierModel model,
  })  : _labels = labels,
        _model = model;

  static Future<Classifier?> loadWith({
    required String labelsFileName,
    required String modelFileName,
  }) async {
    try {
      final labels = await _loadLabels(labelsFileName);
      final model = await _loadModel(modelFileName);
      return Classifier._(labels: labels, model: model);
    } catch (e) {
      debugPrint('Can\'t initialize Classifier: ${e.toString()}');
      if (e is Error) {
        debugPrintStack(stackTrace: e.stackTrace);
      }
      return null;
    }
  }

  static Future<ClassifierModel> _loadModel(String modelFileName) async {
    try {
      debugPrint('Attempting to load model from asset: assets/$modelFileName');

      final options = InterpreterOptions()
        ..threads = 1
        ..useNnApiForAndroid = false; // Disable NNAPI

      final interpreter = await Interpreter.fromAsset(
        'assets/$modelFileName',
        options: options,
      );

      interpreter.allocateTensors(); // Pre-allocate tensors

      debugPrint('Interpreter created successfully');

      final inputShape = interpreter.getInputTensor(0).shape;
      final outputShape = interpreter.getOutputTensor(0).shape;
      final inputType = interpreter.getInputTensor(0).type;
      final outputType = interpreter.getOutputTensor(0).type;

      debugPrint('Input shape: $inputShape');
      debugPrint('Output shape: $outputShape');
      debugPrint('Input type: $inputType');
      debugPrint('Output type: $outputType');

      return ClassifierModel(
        interpreter: interpreter,
        inputShape: inputShape,
        outputShape: outputShape,
        inputType: inputType,
        outputType: outputType,
      );
    } catch (e, stackTrace) {
      debugPrint('Error loading model: $e');
      debugPrint('Stack trace: $stackTrace');
      rethrow;
    }
  }

  static Future<ClassifierLabels> _loadLabels(String labelsFileName) async {
    try {
      final labelsData = await rootBundle.loadString('assets/$labelsFileName');
      final labels = labelsData
          .split('\n')
          .where((label) => label.isNotEmpty)
          .map((label) {
        // Check if the label contains a space
        final spaceIndex = label.indexOf(' ');
        if (spaceIndex == -1) {
          // If no space found, return the whole label
          return label.trim();
        }
        // If space found, remove the index number
        return label.substring(spaceIndex).trim();
      }).toList();

      debugPrint('Labels loaded successfully: $labels');
      return labels;
    } catch (e, stackTrace) {
      debugPrint('Error loading labels: $e');
      debugPrint('Stack trace: $stackTrace');
      rethrow;
    }
  }

  void close() {
    _model.interpreter.close();
  }

  ClassifierCategory predict(img.Image image) {
    debugPrint(
      'Image: ${image.width}x${image.height}, '
      'size: ${image.length} bytes',
    );

    try {
      final processedImageData = _preProcessInput(image);
      debugPrint('Processed data length: ${processedImageData.length}');

      // Create input and output buffers
      final inputBuffer = processedImageData.reshape([1, 224, 224, 3]);

      // Create a 2D list to match the [1, 4] output shape
      final outputBuffer = List.generate(1, (_) => List.filled(4, 0.0));

      debugPrint('Running inference...');
      debugPrint('Model expected input shape: ${_model.inputShape}');
      debugPrint('Model expected output shape: ${_model.outputShape}');

      // Run inference
      _model.interpreter.run(inputBuffer, outputBuffer);

      debugPrint('Raw output: $outputBuffer');

      // Flatten the output and convert to Float32List
      final flattenedOutput = Float32List.fromList(outputBuffer[0]);

      final resultCategories = _postProcessOutput(flattenedOutput);
      final topResult = resultCategories.first;

      debugPrint('Top category: $topResult');

      return topResult;
    } catch (e, stackTrace) {
      debugPrint('Error during inference: $e');
      debugPrint('Stack trace: $stackTrace');
      rethrow;
    }
  }

  List<ClassifierCategory> _postProcessOutput(Float32List outputBuffer) {
    final List<double> probabilities = outputBuffer.toList();
    final categoryList = <ClassifierCategory>[];

    for (var i = 0; i < _labels.length; i++) {
      final category = ClassifierCategory(_labels[i], probabilities[i]);
      categoryList.add(category);
      debugPrint('label: ${category.label}, score: ${category.score}');
    }

    categoryList.sort((a, b) => b.score.compareTo(a.score));
    return categoryList;
  }

  Float32List _preProcessInput(img.Image image) {
    // Crop the image to square
    final minLength = min(image.width, image.height);
    final cropX = (image.width - minLength) ~/ 2;
    final cropY = (image.height - minLength) ~/ 2;
    final croppedImage = img.copyCrop(
      image,
      x: cropX,
      y: cropY,
      width: minLength,
      height: minLength,
    );

    // Resize to expected dimensions (224x224)
    final resizedImage = img.copyResize(
      croppedImage,
      width: 224,
      height: 224,
      interpolation: img.Interpolation.linear,
    );

    // Create buffer with the exact size needed for [1, 224, 224, 3]
    final processedData = Float32List(1 * 224 * 224 * 3);

    // Convert the image to a normalized float array
    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        final pixel = resizedImage.getPixel(x, y);
        final offset = (y * 224 * 3) + (x * 3);

        // Try with [-1, 1] normalization
        processedData[offset] = (pixel.r.toDouble() - 127.5) / 127.5;
        processedData[offset + 1] = (pixel.g.toDouble() - 127.5) / 127.5;
        processedData[offset + 2] = (pixel.b.toDouble() - 127.5) / 127.5;
      }
    }

    return processedData;
  }
}
