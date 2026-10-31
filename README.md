# Breast Cancer Diagnostic Classification Benchmark

An end-to-end machine learning study evaluating five core classification algorithms on clinical biopsy measurements, deployed as an interactive Streamlit web application.

---

## a. Problem Statement
The objective of this project is to develop and evaluate automated classification models capable of predicting whether a breast mass biopsy sample is **Malignant (Class 1)** or **Benign (Class 0)**. Accurate identification of malignancy is crucial in clinical triage to prevent diagnostic delays while minimizing unnecessary invasive procedures for benign cases.

---

## b. Dataset Description
- **Dataset Source:** UCI Machine Learning Repository / Scikit-Learn Breast Cancer Wisconsin (Diagnostic) Dataset.
- **Number of Instances:** 569 patient records (satisfies the assignment requirement of >= 500 rows).
- **Number of Predictors:** 30 continuous clinical features computed from digitized fine needle aspirate (FNA) images (satisfies the requirement of >= 12 features).
- **Target Variable:** Binary diagnostic class (0: Benign, 1: Malignant).
- **Feature Categories:** 10 core morphological characteristics measured across three statistical aggregations (Mean, Standard Error, and Largest/Worst):
  1. Radius
  2. Texture
  3. Perimeter
  4. Area
  5. Smoothness
  6. Compactness
  7. Concavity
  8. Concave points
  9. Symmetry
  10. Fractal dimension
- **Data Preprocessing & Splitting:**
  - Evaluated on an 80-20 stratified train-test split (455 training instances, 114 test holdout samples).
  - Feature normalization was carried out via standard z-score normalization (StandardScaler) on scale-sensitive algorithms (Logistic Regression, KNN, and Support Vector Machine) to prevent feature magnitude bias.

---

## c. GitHub Repository Link
**Repository URL:** https://github.com/amit-padwekar/ml-classification-benchmark

---

## d. Models Used & Performance Comparison

The table below summarizes all 6 required evaluation metrics computed on the unseen test set (114 samples) without data leakage:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Support Vector Machine** | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well across the dataset, tying for the highest test accuracy (98.25%) and achieving the highest overall ROC-AUC (0.9954). With standard scaling applied, the regularized linear decision boundary separates the dense benign and malignant clusters cleanly, exhibiting only 2 misclassifications out of 114 test samples. |
| **Decision Tree** | Yielded the lowest overall accuracy (92.11%) and AUC (0.9163). Because it constructs orthogonal axis-aligned partition boundaries, the tree struggles with high-dimensional feature correlations (like radius, perimeter, and area), leading to slight overfitting despite limiting the tree depth. |
| **KNN** | Achieved solid performance (95.61% accuracy, 0.9054 MCC) using k=5 neighbors. Distance-based classification benefited noticeably from z-score standardization, though local density variations slightly degraded its precision compared to margin-based approaches. |
| **Naive Bayes** | Recorded an accuracy of 93.86% alongside a strong AUC of 0.9878. While the independence assumption is technically violated due to high collinearity among morphological dimensions (e.g., perimeter vs. area), the Gaussian formulation preserved reliable diagnostic ranking. |
| **Support Vector Machine** | Tied with Logistic Regression as the top performer with an accuracy of 98.25%, an F1 score of 0.9861, and an MCC of 0.9623. The Radial Basis Function (RBF) kernel mapped the 30 continuous clinical features into a higher-dimensional space where maximum-margin separation was easily established. |
| **Overall Winner for your dataset?** | **Logistic Regression** (closely matched by SVM). While both models achieved identical accuracy (0.9825), Precision (0.9861), Recall (0.9861), and MCC (0.9623), Logistic Regression holds a slightly higher ROC-AUC (0.9954 vs. 0.9950) and provides direct probabilistic interpretability without kernel computation overhead. |

---

## Project Structure
```text
ml_assignment_2/
├── app.py                 # Streamlit web application
├── train_and_eval.py      # Pipeline script for model training and metric logging
├── requirements.txt       # Environment dependencies
├── README.md              # Project documentation and metric comparisons
├── test_data.csv          # 20% holdout test data for evaluation upload
└── model/                 # Serialized model artifacts
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── svm.pkl

