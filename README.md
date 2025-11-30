<img width="1896" height="704" alt="image_original" src="https://github.com/user-attachments/assets/67e13594-f690-4ab6-a06e-063eb64eb417" /># Hybrid-Mushroom-Classification-System
A production-ready deep learning system for multi-class mushroom species identification, combining transfer learning (ResNet50) with traditional computer vision techniques to achieve 95%+ accuracy.
🎯 Project Overview
This project implements a hybrid machine learning pipeline that fuses deep features from pre-trained ResNet50 with handcrafted computer vision features (GLCM, HOG, LBP) for robust mushroom species classification. The system includes comprehensive image preprocessing, automated feature extraction, and ensemble learning with real-time inference capabilities.
Key Features

✅ 95%+ Classification Accuracy using ensemble voting
✅ Hybrid Feature Extraction combining deep learning + traditional CV
✅ Advanced Image Preprocessing with background removal and augmentation
✅ Production Deployment via Gradio web application
✅ Real-time Inference with confidence scoring

🏗️ Architecture
Input Image
    ↓
[Preprocessing Pipeline]
    ├─ HSV Background Removal
    ├─ Otsu Thresholding
    ├─ Brightness Normalization
    └─ Data Augmentation (rotation, flipping)
    ↓
[Feature Extraction]
    ├─ Deep Features: ResNet50 (2048 dims)
    └─ Handcrafted Features:
        ├─ GLCM (texture analysis)
        ├─ HOG (edge detection)
        ├─ LBP (pattern recognition)
        └─ Color statistics (HSV, LAB)
    ↓
[Feature Processing]
    ├─ StandardScaler normalization
    ├─ SMOTE class balancing
    ├─ PCA dimensionality reduction (2048→100)
    ├─ Polynomial feature expansion
    └─ Mutual information feature selection
    ↓
[Ensemble Classifier]
    ├─ Random Forest
    ├─ Support Vector Machine
    └─ Logistic Regression
    ↓
Prediction + Confidence Score
🚀 Quick Start
Prerequisites
bashPython 3.8+
TensorFlow 2.x
OpenCV
scikit-learn
scikit-image
imbalanced-learn
gradio

Installation
bash# Clone repository
git clone https://github.com/Sowmya721/mushroom-classification.git
cd mushroom-classification

# Install dependencies
pip install -r requirements.txt

Usage
Training the Model:
pythonpython train_model.py --data_path ./Mushrooms --epochs 50

Running Web Application:
pythonpython app.py
Access the application at http://localhost:7860
Making Predictions:
pythonfrom model import MushroomClassifier

classifier = MushroomClassifier()
classifier.load_model('models/ensemble_model.pkl')

prediction, confidence = classifier.predict('path/to/mushroom_image.jpg')
print(f"Species: {prediction}, Confidence: {confidence:.2%}")


📊 Performance Metrics
ModelAccuracyPrecisionRecallF1-ScoreRandom Forest93.7%0.940.930.93SVM92.4%0.920.920.92Logistic Regression89.8%0.900.890.89Ensemble (Voting)95.3%0.950.950.95
Training Results

Dataset Size: 175 samples across 4 species
Training Time: ~10 minutes on GPU
Model Size: 85MB (ensemble)
Inference Time: ~200ms per image

🛠️ Technical Details
Feature Extraction
Deep Features (ResNet50):

Pre-trained on ImageNet
Global Average Pooling
2048-dimensional feature vector

Handcrafted Features:

GLCM: Texture features (contrast, homogeneity, energy, correlation)
HOG: Edge detection with 9 orientations, 16×16 cells
LBP: Local binary patterns (8 neighbors, radius 1)
Color Statistics: Mean, std, skewness, kurtosis in HSV and LAB spaces

Data Preprocessing

HSV-based background removal for isolation
Otsu automatic thresholding for segmentation
Adaptive brightness normalization
Data augmentation: rotation (±20°), horizontal flipping, brightness adjustment

Dimensionality Reduction

PCA: 2048 → 100 principal components
Polynomial features: degree 2
Feature selection: Top 120 features via mutual information


🎨 Web Application
The Gradio interface provides:

Drag-and-drop image upload
Real-time species prediction
Confidence scores for all classes
Top-3 predictions display
Preprocessing visualization
<img width="1896" height="704" alt="image_original" src="https://github.com/user-attachments/assets/04f1568f-3904-467f-a345-00f8c6fa2e73" />

 Implement YOLO for automatic mushroom detection in natural scenes
 Add ONNX export for cross-platform deployment
 Integrate attention mechanisms for interpretability
 Expand the dataset to include more species
 Add mobile app deployment (TensorFlow Lite)
 Implement test-time augmentation for improved accuracy

📈 Results Visualization
Training curves and prediction comparisons available in /results:

Model performance comparison
Confusion matrices
Feature importance analysis
Sample predictions with confidence


👤 Author
Sowmya Negi

GitHub: @Sowmya721
LinkedIn: sowmya-negi
Email: ssnalm987@gmail.com

🙏 Acknowledgments

ResNet50 architecture from Keras Applications
Mushroom dataset from [source]
Inspired by research in hybrid feature learning for image classification
