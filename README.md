# Digit Classification using KNN

This project classifies handwritten digits (0-9) using the K-Nearest Neighbors (KNN) algorithm. The goal is to find the value of K that maximizes the classification score. Additionally, the project includes plotting the confusion matrix and the classification report.

## Requirements

- scikit-learn
- matplotlib
- seaborn
- pandas
- joblib

## Dataset

The dataset used is the `digits` dataset from `sklearn.datasets`, which contains 8x8 images of handwritten digits.

## Steps

1. **Load Dataset**: Load the digits dataset using `load_digits()` from `sklearn.datasets`.
2. **Data Split**: Split the data into training and testing sets using `train_test_split()` from `sklearn.model_selection`.
3. **KNN Model**: Use the KNN classifier from `sklearn.neighbors` to classify digits. Perform a randomized search to find the optimal value of K.
4. **Confusion Matrix**: Plot the confusion matrix using `confusion_matrix` from `sklearn.metrics` and visualize it using `seaborn`.
5. **Classification Report**: Generate and print the classification report using `classification_report` from `sklearn.metrics`.
