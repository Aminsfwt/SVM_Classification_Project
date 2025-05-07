# Heart Disease Prediction Using SVM

This repository contains a machine learning project for heart disease prediction using Support Vector Machines (SVM). 
The project includes data preprocessing, feature scaling, model training, evaluation, and hyperparameter tuning using GridSearchCV.
The code is implemented in Python using popular libraries such as scikit-learn, Pandas, NumPy, and Matplotlib.

## Repository Structure

```
Applied ML/
└── Linear Model For Classification/
    └── Heart Disease Prediction/
        ├── heart.csv                   # Dataset file
        ├── main_svm.py                 # Main script for SVM training and evaluation
        └── PrepData.py                 # Utility functions for data reading and preprocessing
```

## Overview

- **Data Reading and Preprocessing:**  
  The script reads the heart disease dataset from `heart.csv`, cleans the data by dropping missing values, and splits it into training and testing sets.

- **Feature Scaling:**  
  Features are scaled using `StandardScaler` on the training data (and applied to the test data) for better model performance.

- **Model Training and Evaluation:**  
  A linear SVM model (`LinearSVC`) is trained; predictions and model accuracy are printed. A confusion matrix is also generated for model evaluation.

- **Hyperparameter Tuning:**  
  `GridSearchCV` is used with `SVC` to search for the best hyperparameters based on accuracy. For linear kernels, the project also displays feature importance based on the model coefficients.

## Requirements

- Python 3.x
- Pandas
- NumPy
- scikit-learn
- Matplotlib

You can install the required packages using:

````bash
pip install pandas numpy scikit-learn seaborn matplotlib
