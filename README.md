# 🧠 Parkinson's Disease Prediction Web App

This repository contains a **Flask-based web application** for predicting Parkinson's disease using machine learning models. The application allows users to input medical measurements and receive predictions based on trained models.

## 🚀 Features
- **Machine Learning Models:** SVM, Decision Tree, Random Forest, Gradient Boosting
- **Web Interface:** User-friendly input form for predictions
- **Model Training & Evaluation:** Comprehensive training script with performance metrics
- **Data Preprocessing:** Feature scaling and class imbalance handling using SMOTEENN
- **Visualizations:** Heatmaps, ROC curves, precision-recall curves

## 📂 Repository Structure

```
📦 parkinsons-disease-prediction
├── 📄 README.md               # Project Overview
├── 📄 app.py                  # Flask Web App
├── 📄 parkinsons_disease_detection.py # Model Training Script
├── 📄 requirements.txt         # Dependencies
├── 📂 templates/
│   ├── index.html            # Web Interface Template
├── 📂 models/
│   ├── svm_model.pkl         # Saved SVM Model
│   ├── dt_model.pkl          # Saved Decision Tree Model
│   ├── rf_model.pkl          # Saved Random Forest Model
│   ├── gb_model.pkl          # Saved Gradient Boosting Model
│   ├── scaler.pkl            # Saved Scaler for Preprocessing
├── 📂 data/
│   ├── parkinsons.csv        # Dataset
```

## 🏗️ Setup Instructions

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/parkinsons-disease-prediction.git
cd parkinsons-disease-prediction
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Web Application
```sh
python app.py
```

📌 The application will be available at **http://127.0.0.1:5000/**

## 🔬 Model Training & Evaluation
The `parkinsons_disease_detection.py` script:
- Loads the dataset (`parkinsons.csv`)
- Performs **Exploratory Data Analysis (EDA)**
- Trains models (**SVM, Decision Tree, Random Forest, Gradient Boosting**)
- Evaluates models using accuracy, confusion matrix, and ROC curves
- Saves the best-performing models

## 📊 Visualizations
The script includes:
- **Feature correlations** using heatmaps
- **Model comparison plots**
- **ROC & Precision-Recall curves**

## 📌 Usage
1. **Run the Flask app** and open it in a browser.
2. **Enter medical measurements** in the form.
3. **Submit** to get a prediction on Parkinson’s disease.
4. **Example data** is provided for testing.

## ⚖️ License
This project is licensed under the **MIT License**. Feel free to use and modify it.

---
🚀 **Contributions are welcome!** Feel free to fork this repo and improve the model or UI. Happy coding! 🎯
