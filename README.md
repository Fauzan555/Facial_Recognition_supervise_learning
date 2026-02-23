# Facial_Recognition_using supervise_learning

# 📌 Project Overview

This project presents a supervised machine learning based facial recognition system designed to identify and safeguard high profile individuals. The objective is to accurately distinguish Arnold Schwarzenegger from the general population using structured facial feature representations derived from PCA.

The system evaluates multiple classification algorithms through cross validation and selects the best performing model for final deployment. The final pipeline achieves strong accuracy while maintaining model simplicity and interpretability.

# Problem Statement

In security sensitive environments, the ability to reliably detect and verify the identity of high visibility individuals is critical. Variations in lighting, pose, and image quality make this task non trivial.

This project builds a binary classification model to determine whether a given face belongs to Arnold Schwarzenegger or not, using supervised learning techniques on structured facial embeddings.

# Dataset Description

The dataset is derived from the Labeled Faces in the Wild dataset and preprocessed into structured numerical features.

File: data/lfw_arnie_nonarnie.csv

# Dataset Composition:

40 images of Arnold Schwarzenegger.

150 images of other individuals.

Total: 190 samples.

# Feature Engineering:

PCA applied to facial images

Columns: PC1, PC2, ..., PCN

Label:

1 → Arnold Schwarzenegger

0 → Other individuals

This dimensionality reduction approach captures the most informative facial characteristics while reducing computational complexity.

# Tech Stack

Python

Pandas

Scikit Learn

NumPy

Matplotlib

Seaborn

# Methodology
# 1️⃣ Data Preparation

Loaded PCA transformed dataset

Separated predictors and target label

Applied stratified train test split to maintain class balance

# 2️⃣ Model Development

Constructed machine learning pipelines for:

Logistic Regression

Random Forest

Support Vector Machine

Each model was evaluated using 5 fold cross validation to ensure robustness and prevent overfitting.

# 3️⃣ Model Selection

The best performing model based on mean cross validation accuracy was selected.

# 🏆 Selected Model: Logistic Regression
📊 Cross Validation Accuracy: 0.822

# 4️⃣ Final Evaluation on Test Set
Metric	Score
Accuracy	0.8158
Precision	1.00
Recall	0.125
F1 Score	0.222

# Key Insights

Logistic Regression performed best among the evaluated models.

High precision indicates strong confidence when predicting Arnold.

Low recall suggests the model is conservative and misses some positive instances.

Class imbalance impacts recall performance.

# 🚀 What This Demonstrates

This project highlights:

End to end ML pipeline construction

Model comparison using cross validation

Structured feature engineering using PCA

Proper evaluation using multiple performance metrics

Handling imbalanced classification challenges

# 🔮 Future Improvements

Apply class balancing techniques such as SMOTE

Tune hyperparameters using GridSearchCV

Use deep learning based embeddings such as FaceNet

Expand dataset size for improved generalization

Deploy as a real time API

# Repository Structure
├── data/
├── notebook.ipynb
├── README.md
└── requirements.txt

