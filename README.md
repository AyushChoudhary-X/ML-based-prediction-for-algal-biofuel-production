# 🔬 Algal Digestion ML App

A complete end-to-end Machine Learning web application built using **Flask**, designed to analyze, compare, and optimize models for predicting **methane yield / biogas production**.

This app allows users to upload their dataset, train multiple ML models, visualize performance, interpret results, and even find optimal input conditions.

---

## 🚀 Features

### 📊 Model Training & Comparison
- Train multiple ML models:
  - Random Forest
  - SVR
  - Linear Regression
  - KNN
  - XGBoost
  - ANN
- Automatic performance evaluation using:
  - R² Score
  - RMSE
- Best model selection

---

### 📈 Interactive Dashboard
- Plotly-based interactive graphs:
  - Model comparison (R² & RMSE)
  - Actual vs Predicted scatter plot
- Hover, zoom, and explore data dynamically

---

### 🔍 Feature Importance
- Visualize feature importance for supported models
- Helps understand which inputs influence output most

---

### 🧠 SHAP Explainability
- Model interpretability using SHAP
- Global feature impact visualization
- Useful for research and viva explanations

---

### ⚙️ Optimization Module
- Finds optimal input conditions for:
  - Maximizing output
  - Minimizing output
- Uses randomized search over feature space

---

### 🔮 Prediction Interface
- Manual input for prediction
- Automatic handling of missing values
- Uses best trained model

---

### 📥 Download Results
- Download full dataset including:
  - Input features
  - Actual values
  - Predictions from all models

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Plotly, Matplotlib
- **Explainability:** SHAP
- **Frontend:** HTML, CSS, Jinja2

---

## 📌 How It Works

1. Upload your CSV dataset
2. Select target variable
3. Choose ML models
4. Train models
5. View dashboard & compare performance
6. Explore feature importance & SHAP
7. Optimize inputs
8. Make predictions
