# Pneumonia Detection using Chest X-ray Images

This project implements a deep learning model to detect pneumonia from chest X-ray images using a DenseNet121-based Convolutional Neural Network (CNN).  
Grad-CAM visualization is used to highlight the lung regions that most strongly influence the model’s predictions, improving interpretability.

## Features
- Binary classification: NORMAL vs PNEUMONIA
- Transfer learning with DenseNet121
- Grad-CAM visualization for explainability
- Highlighted pneumonia-suspected regions

## Sample Result
![Result](results/pneumonia_detection_result.png)

## Model Pipeline
1. Data loading from CSV
2. Preprocessing with lung-focused cropping
3. CNN training
4. Model evaluation
5. Grad-CAM visualization

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Matplotlib
- Scikit-learn
