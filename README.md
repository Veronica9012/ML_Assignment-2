# Machine Learning Assignment 2 - BITS Pilani WILP

## a. Problem Statement
The objective of this assignment is to implement, evaluate, and compare five standard classification machine learning algorithms on a public dataset. Models are trained offline and deployed via an interactive Streamlit web application on Streamlit Community Cloud for real-time model performance evaluation.

## b. Dataset Description
- **Source**: UCI Machine Learning Repository (Wine Quality Dataset - Red Wine)
- **Total Instances**: 1,599 instances
- **Total Features**: 12 columns (11 physical/chemical input features + 1 target variable)
- **Input Features**: Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol
- **Target Variable**: Binary classification (1 = Quality >= 6, 0 = Quality < 6)

## c. GitHub Repository Link
https://github.com/Veronica9012/ML_Assignment-2.git

## d. Live Streamlit App Link
https://mlassignment-2-k7abn7suvwygs7pkkkzse4.streamlit.app

## e. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7406 | 0.8242 | 0.7683 | 0.7368 | 0.7522 | 0.4808 |
| **Decision Tree** | 0.7281 | 0.7842 | 0.7303 | 0.7778 | 0.7533 | 0.4501 |
| **KNN** | 0.7375 | 0.7963 | 0.7584 | 0.7427 | 0.7505 | 0.4735 |
| **Naive Bayes** | 0.7219 | 0.7911 | 0.7253 | 0.7719 | 0.7479 | 0.4376 |
| **Support Vector Machine** | 0.7594 | 0.8248 | 0.7639 | 0.8012 | 0.7821 | 0.5140 |

## f. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Showed reliable decision boundaries across features; benefits strongly from feature scaling. |
| **Decision Tree** | Handled non-linear patterns, but slight overfitting restricted its test accuracy compared to linear classifiers. |
| **KNN** | Moderate performance at K=5; sensitive to distance metrics across correlated features. |
| **Naive Bayes** | Extremely fast execution, though independence assumptions between chemical properties reduced overall precision. |
| **Support Vector Machine** | Best overall performer; RBF kernel captured complex feature interactions effectively. |
| **Overall Winner** | **Support Vector Machine (SVM)** achieved the highest performance across Accuracy (0.7594), AUC (0.8248), and MCC (0.5140). |
