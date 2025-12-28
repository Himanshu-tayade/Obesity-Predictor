# Predicting Obesity Level using Machine Learning and Deep Learning

## Overview

This project was developed as part of the **Infosys Springboard 5.0 Internship**. The goal was to predict obesity levels using machine learning and deep learning techniques. The team collaborated to preprocess data, perform exploratory data analysis (EDA), and train multiple classification models, achieving competitive performance using both traditional machine learning models and the TabNet neural network.

- **Role**: Model Development & UI (7-member team)
- **Technologies Used**: Python, Scikit-learn, TabNet, MLflow, Pandas, NumPy, Matplotlib/Seaborn (for EDA)
- **Dataset**: Kaggle obesity dataset with synthetic feature enhancement.


## Project Objectives

- Develop a classification model to predict obesity levels based on input features.
- Compare performance of machine learning (SVM, Random Forest, Logistic Regression) and deep learning (TabNet) models.
- Optimize model performance through hyperparameter tuning and track experiments using MLflow.

## Methodology

### 1. Data Preprocessing & EDA

- **Preprocessing**: Cleaned dataset, handled missing values, and performed feature engineering (e.g., encoding categorical variables, scaling numerical features).
- **EDA**: Conducted exploratory data analysis to identify key patterns and correlations.
- Tools: Pandas, NumPy, Matplotlib, Seaborn.

### 2. Model Development

- **Models Evaluated**:
  - **Machine Learning**: Support Vector Machine (SVM), Random Forest, Logistic Regression.
  - **Deep Learning**: TabNet (a neural network-based model designed for tabular data).
- **Hyperparameter Tuning**: Used MLflow to track and visualize model performance across various hyperparameter configurations (e.g., learning rate, batch size for TabNet).

## Model Performance Comparison

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | 89.62%   |
| Random Forest       | 100.00%  |
| SVM                 | 99.82%   |
| XGBoost             | 99.75%   |
| Deep Learning       | 99.25%   |
| TabNet              | 99.38%   |

> Note: Several models achieved very high accuracy (>99%). This may indicate strong feature separability in the dataset or potential overfitting. Further validation using cross-validation and additional metrics is recommended.


### 3. Results

- *Key Insight*: While several traditional machine learning models achieved very high accuracy, TabNet demonstrated strong and consistent performance as a deep learning model designed specifically for tabular data.


## Contributors

- Ankit Sharma
- Abhishek Mane
- Himanshu Tayade
- Manvitha
- Ankita Gupta
- Yamini
- Archana Reddy 
