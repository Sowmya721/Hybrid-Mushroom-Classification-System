# Hybrid-Mushroom-Classification-System
Fully automated AI system capable of identifying mushroom types from images. 
This project implements a hybrid mushroom classification system using both handcrafted image features (GLCM, HOG, LBP, color statistics) and deep feature extraction with ResNet50. 
The extracted features are combined, optimized using scaling, SMOTE balancing, PCA, polynomial feature expansion, and mutual information-based feature selection. 
Multiple machine learning models are trained and compared, and finally, a soft-voting ensemble delivers the best performance for high-accuracy mushroom species prediction.
