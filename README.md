# Fine-Grained Bird Classification

This repository contains the implementation of Task 2 of the Deep Learning course (CSC8637), focusing on fine-grained bird classification using the CUB-200-2011 dataset(https://data.caltech.edu/records/65de6-vp158). The project utilizes transfer learning with EfficientNet-B3 to classify 200 bird species.

## Project Overview

The goal of this project was to develop a deep learning model capable of classifying bird species in a fine-grained manner using the CUB-200-2011 dataset. Transfer learning was employed using the EfficientNet-B3 model, which was fine-tuned to predict bird species based on images.

### Key Steps:
1. **Data Preprocessing & Augmentation**: The dataset was split into training, validation, and test sets, and augmented to improve model generalization.
2. **Model Selection & Training**: EfficientNet-B3 was selected for transfer learning, with hyperparameter tuning for optimization.
3. **Evaluation Metrics**: The model's performance was evaluated based on accuracy, precision, recall, F1-score, and confusion matrix.

## Dataset

The CUB-200-2011 dataset contains images of 200 bird species. It was split into:
- **Training Images**: 5994
- **Test Images**: 5794
- **Training Samples After Split**: 5400
- **Validation Samples**: 594

The images were resized to 224x224 pixels, and data augmentation techniques such as rotation, zooming, and horizontal flipping were applied to improve model robustness.

## Methodology

### Selected Model: EfficientNet-B3
After evaluating several models (ResNet50, VGG16, Xception), EfficientNet-B3 was chosen due to its efficiency and accuracy for fine-grained classification tasks.

### Hyperparameters:
- Optimizer: Adam
- Learning Rate: 0.0005
- Batch Size: 32
- Loss Function: Cross-Entropy Loss
- Epochs: 50 (early stopping at epoch 33)

### Training Results (Epoch 33):
- Training Accuracy: 96.65%
- Validation Accuracy: 77.78%
- Test Accuracy: 76.63%

### Performance Metrics:
- **Test Precision (Macro)**: 78.02%
- **Test Recall (Macro)**: 76.88%
- **Test F1-score (Macro)**: 76.85%

## Model Improvements

Possible improvements to enhance model performance include:
- Enhanced data augmentation (brightness, contrast adjustments, etc.)
- Combining models through ensemble learning
- Fine-tuning additional layers
- Increasing image resolution
- Hyperparameter optimization using methods like Bayesian Optimization

## Future Work

The current model achieved a test accuracy of 76.63%, but further improvements are needed to reach an accuracy of over 80%. Future work will focus on optimizing hyperparameters, applying more advanced data augmentations, and increasing the image resolution.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- Numpy
- Matplotlib
- Scikit-learn

