# Documentation — `lib/classifier/classifier.dart`

**Application :** Plant Recognizer (Flutter)
**Audience :** Étudiants BUT Informatique 3ème année
**Sujet :** Reconnaissance de plantes par modèle TensorFlow Lite embarqué

---

## 1. Vue d'ensemble

Le fichier `classifier.dart` contient la classe `Classifier`, qui orchestre toute la chaîne d'inférence d'un modèle TensorFlow Lite (TFLite) : chargement du modèle, prétraitement de l'image, exécution de l'inférence, et post-traitement des résultats.

### Fichiers liés

| Fichier | Rôle |
|---|---|
| `classifier_model.dart` | Encapsule l'interpréteur TFLite et les métadonnées du modèle (shapes, types) |
| `classifier_category.dart` | Structure de données simple : `label` (String) + `score` (double) |

---

## 2. Structure générale de la classe

### 2.1 Attributs privés

```dart
final ClassifierLabels _labels;  // Liste des noms de catégories
final ClassifierModel _model;    // Modèle TFLite chargé
```

Les attributs sont `final` (immuables) et préfixés `_` (privés, inaccessibles de l'extérieur).

### 2.2 Constructeur privé et pattern Factory

```dart
Classifier._({ required ClassifierLabels labels, required ClassifierModel model })
  : _labels = labels, _model = model;
```

Le constructeur est **privé** (préfixe `_`). L'instanciation est forcée via la méthode statique `loadWith()`.

> **Pourquoi ?** Le chargement du modèle est asynchrone. Un constructeur Dart ne peut pas être `async`, donc on délègue la construction à une méthode statique qui retourne un `Future`.

### 2.3 Chargement — `loadWith()`

```dart
static Future<Classifier?> loadWith({
  required String labelsFileName,
  required String modelFileName,
}) async { ... }
```

- Appelle `_loadLabels()` : lit un fichier texte d'assets (une étiquette par ligne).
- Appelle `_loadModel()` : charge le fichier `.tflite`, configure l'interpréteur, pré-alloue les tenseurs.
- Retourne `null` en cas d'échec (le `?` dans `Classifier?`).

### 2.4 Libération des ressources — `close()`

```dart
void close() {
  _model.interpreter.close();
}
```

Libère la mémoire native allouée par TFLite. À appeler lorsque le `Classifier` n'est plus utilisé (bonne pratique sur mobile).

---

## 3. La méthode `predict()` — Analyse détaillée

C'est le **cœur fonctionnel** du classifier. Elle prend une image en entrée et retourne la catégorie de plante la plus probable.

```dart
ClassifierCategory predict(img.Image image)
```

### 3.1 Vue d'ensemble du pipeline

```
Image brute (quelconque)
        │
        ▼
 ┌─────────────────┐
 │ _preProcessInput │  → Crop + Resize 224×224 + Normalisation [-1,1]
 └─────────────────┘
        │
        ▼  Float32List [150528 valeurs]
        │
        ▼
 ┌──────────────┐
 │   reshape    │  → [1, 224, 224, 3]  (format attendu par le modèle)
 └──────────────┘
        │
        ▼
 ┌───────────────────┐
 │ interpreter.run() │  → Inférence TFLite
 └───────────────────┘
        │
        ▼  outputBuffer [1][4]  (4 probabilités brutes)
        │
        ▼
 ┌──────────────────────┐
 │ _postProcessOutput() │  → Association label + score, tri décroissant
 └──────────────────────┘
        │
        ▼
 ClassifierCategory (meilleur résultat)
```

---

### 3.2 Étape 1 — Prétraitement : `_preProcessInput()`

```dart
Float32List _preProcessInput(img.Image image)
```

#### Recadrage carré centré

```dart
final minLength = min(image.width, image.height);
final cropX = (image.width - minLength) ~/ 2;
final cropY = (image.height - minLength) ~/ 2;
final croppedImage = img.copyCrop(image, x: cropX, y: cropY,
                                  width: minLength, height: minLength);
```

Une photo prise en mode portrait (ex. 1080×1920) ou paysage n'est pas carrée. Le modèle TFLite attend une image **carrée**. On extrait donc le plus grand carré possible centré dans l'image originale.

> L'opérateur `~/` effectue une **division entière** en Dart (équivalent de `Math.floor(a/b)`).

#### Redimensionnement à 224×224

```dart
final resizedImage = img.copyResize(croppedImage,
    width: 224, height: 224,
    interpolation: img.Interpolation.linear);
```

Le modèle MobileNet sous-jacent a été entraîné sur des images **exactement 224×224 pixels**. L'interpolation linéaire recalcule les valeurs des pixels intermédiaires lors du redimensionnement pour préserver la qualité visuelle.

#### Normalisation des valeurs de pixels

```dart
final processedData = Float32List(1 * 224 * 224 * 3);  // 150 528 floats

for (var y = 0; y < 224; y++) {
  for (var x = 0; x < 224; x++) {
    final pixel = resizedImage.getPixel(x, y);
    final offset = (y * 224 * 3) + (x * 3);

    processedData[offset]     = (pixel.r.toDouble() - 127.5) / 127.5;  // R
    processedData[offset + 1] = (pixel.g.toDouble() - 127.5) / 127.5;  // G
    processedData[offset + 2] = (pixel.b.toDouble() - 127.5) / 127.5;  // B
  }
}
```

**Pourquoi normaliser ?**
Les valeurs brutes de pixels sont des entiers entre 0 et 255. Les réseaux de neurones ont des performances nettement meilleures lorsque leurs entrées sont proches de zéro, avec une variance faible. On applique la formule :

```
valeur_normalisée = (valeur_brute - 127.5) / 127.5
```

Ce qui ramène chaque composante dans l'intervalle **[-1.0 ; +1.0]**. C'est la normalisation standard pour les modèles de type **MobileNet**.

**Layout mémoire (format HWC) :**
Les canaux R, G, B de chaque pixel sont stockés de façon entrelacée :

```
[R(0,0), G(0,0), B(0,0),  R(1,0), G(1,0), B(1,0),  ...,  R(223,223), G(223,223), B(223,223)]
   offset=0                  offset=3                          offset=150525
```

Le calcul d'offset `(y * 224 * 3) + (x * 3)` positionne correctement chaque pixel dans ce tableau linéaire.

---

### 3.3 Étape 2 — Reshape et inférence

```dart
final inputBuffer = processedImageData.reshape([1, 224, 224, 3]);
final outputBuffer = List.generate(1, (_) => List.filled(4, 0.0));

_model.interpreter.run(inputBuffer, outputBuffer);
```

**Le reshape** reformate le tableau plat de 150 528 valeurs en tenseur 4D de shape `[1, 224, 224, 3]` :

| Dimension | Valeur | Signification |
|---|---|---|
| `1` | batch size | On traite 1 image à la fois |
| `224` | height | Hauteur en pixels |
| `224` | width | Largeur en pixels |
| `3` | channels | R, G, B |

**L'outputBuffer** est pré-dimensionné `[1][4]` : 1 résultat de batch, 4 scores (un par catégorie de plante reconnue par le modèle).

**`interpreter.run()`** exécute le modèle TFLite en mode synchrone : le réseau de neurones calcule les probabilités d'appartenance à chacune des 4 catégories et écrit le résultat dans `outputBuffer`.

---

### 3.4 Étape 3 — Post-traitement : `_postProcessOutput()`

```dart
List<ClassifierCategory> _postProcessOutput(Float32List outputBuffer) {
  final categoryList = <ClassifierCategory>[];

  for (var i = 0; i < _labels.length; i++) {
    final category = ClassifierCategory(_labels[i], probabilities[i]);
    categoryList.add(category);
  }

  categoryList.sort((a, b) => b.score.compareTo(a.score));
  return categoryList;
}
```

- Associe chaque score (index `i` du buffer de sortie) à son étiquette correspondante (index `i` de `_labels`).
- Trie la liste par score **décroissant** : `b.score.compareTo(a.score)` place les scores les plus élevés en premier.
- `predict()` retourne ensuite `resultCategories.first` — la catégorie avec la probabilité la plus haute.

---

## 4. Schéma récapitulatif complet

```
┌──────────────────────────────────────────────────────────────┐
│                        Classifier                            │
│                                                              │
│  loadWith(labelsFileName, modelFileName)                     │
│    ├─ _loadLabels()  →  ["Rose", "Tulipe", "Tournesol", ...] │
│    └─ _loadModel()   →  Interpreter TFLite (tenseurs alloués)│
│                                                              │
│  predict(image)                                              │
│    ├─ _preProcessInput(image)                                │
│    │    ├─ Crop carré centré        (img.copyCrop)           │
│    │    ├─ Resize 224×224           (img.copyResize)         │
│    │    └─ Normalisation [-1, 1]  →  Float32List[150528]     │
│    │                                                         │
│    ├─ reshape([1, 224, 224, 3])                              │
│    ├─ interpreter.run()          →  outputBuffer[1][4]       │
│    └─ _postProcessOutput()       →  ClassifierCategory       │
│                                                              │
│  close()  →  libère la mémoire native TFLite                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Points clés à retenir

| Concept | Application dans le code |
|---|---|
| **Pattern Factory** | `loadWith()` + constructeur `_` privé |
| **Async/Await** | `Future<Classifier?>`, `Future<ClassifierModel>` |
| **Gestion d'erreurs** | `try/catch` avec retour `null` ou `rethrow` |
| **Normalisation ML** | Pixels [0–255] → [-1, 1] avant inférence |
| **Shape de tenseur** | `[batch, height, width, channels]` = `[1, 224, 224, 3]` |
| **Layout HWC** | Canaux R, G, B entrelacés pixel par pixel |
| **Tri fonctionnel** | `list.sort((a, b) => b.score.compareTo(a.score))` |
| **Gestion mémoire** | `close()` libère les ressources natives TFLite |
