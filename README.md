# Loan Amount Prediction Using Machine Learning
**Production-Grade Regression System & Streamlit Web Deployment**

[![GitHub Repository](https://img.shields.io/badge/GitHub-Loan--Amount--Prediction--ML-blue?logo=github)](https://github.com/harshal25-hub/Loan-Amount-Prediction-ML)
[![Streamlit App](https://img.shields.io/badge/Deployment-Streamlit-FF4B4B?logo=streamlit)](http://127.0.0.1:8501)
[![Dataset](https://img.shields.io/badge/Dataset-Real%20Online%20Competition%20Data-brightgreen)](#-dataset)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://python.org)

---

## 🌟 Project Overview
This repository contains an end-to-end Machine Learning system developed to predict sanctioned loan amounts for prospective applicants in a financial institution. Trained on a **real-world online competition dataset (30,000 records from HackerEarth / Kaggle)**, the system evaluates applicant incomes, credit scores (CIBIL), existing debt liabilities, requested tenure, and collateral asset valuations to provide real-time loan underwriting estimates in Indian Rupees (**₹**).

The champion model, **Gradient Boosting Regression**, achieved an **$R^2$ score of 0.9888** and the lowest **RMSE of ₹3,91,191.33**, outperforming Linear Regression, Decision Tree Regression, and Random Forest Regression.

---

## 📊 4-Model Comparative Study & Benchmark

All 4 algorithms were trained with scikit-learn preprocessing pipelines (`StandardScaler` + `OneHotEncoder`) and benchmarked on an unseen 20% holdout test set:

| Rank | Machine Learning Algorithm | MAE (₹) | MSE (₹²) | RMSE (₹) | Test $R^2$ Score | Train $R^2$ | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Gradient Boosting Regression** | **₹2,50,815.84** | **1.53 × 10¹¹** | **₹3,91,191.33** | **0.9888** | **0.9948** | **Optimal Champion** |
| **#2** | **Random Forest Regression** | ₹2,58,950.02 | 1.60 × 10¹¹ | ₹3,99,623.06 | 0.9883 | 0.9938 | Robust Ensemble |
| **#3** | **Decision Tree Regression** | ₹2,74,960.44 | 1.89 × 10¹¹ | ₹4,34,535.01 | 0.9861 | 0.9900 | Step Non-Linearity |
| **#4** | **Linear Regression** | ₹3,04,394.08 | 2.08 × 10¹¹ | ₹4,56,024.85 | 0.9847 | 0.9864 | Fast Baseline |

---

## 📁 Repository Structure

```
├── app.py                           # Interactive Streamlit Web Application
├── requirements.txt                 # Dependencies for local and Streamlit Cloud deployment
├── data/
│   ├── real_loan_dataset.csv        # Authentic 30,000-record online competition dataset
│   ├── process_online_dataset.py    # Cleaning, imputation & currency conversion pipeline
│   └── loan_data.csv                # 10,000-record clean production dataset
├── models/
│   ├── best_model.joblib            # Champion Gradient Boosting Pipeline
│   ├── gradient_boosting_regression.joblib
│   ├── random_forest_regression.joblib
│   ├── decision_tree_regression.joblib
│   ├── linear_regression.joblib
│   ├── metrics_comparison.json      # Structured evaluation metrics
│   └── model_metadata.json          # Preprocessing & categorical feature schema
├── src/
│   ├── eda.py                       # Exploratory Data Analysis & visual plot generator
│   ├── train_models.py              # Pipelines, model training, evaluation & metrics
│   ├── predict.py                   # Reusable inference engine & underwriter metrics
│   └── convert_to_pdf.py            # ReportLab script converting markdown docs to PDF
├── plots/                           # 10 High-Resolution Analysis & Evaluation Charts
│   ├── correlation_heatmap.png
│   ├── loan_amount_distribution.png
│   ├── request_vs_sanctioned.png
│   ├── credit_score_impact.png
│   ├── income_asset_impact.png
│   ├── location_defaults_impact.png
│   ├── model_comparison_metrics.png
│   ├── actual_vs_predicted.png
│   ├── residuals_distribution.png
│   └── feature_importance.png
├── ml_topics_explanation.pdf        # Doc 1: Comprehensive ML Topics (PDF)
├── project_files_explanation.pdf    # Doc 2: Comprehensive Project Files (PDF)
├── ml_models_explanation.pdf        # Doc 3: Comprehensive ML Models (PDF)
├── ml_topics_explanation.md         # Doc 1: Markdown Source
├── project_files_explanation.md     # Doc 2: Markdown Source
├── ml_models_explanation.md         # Doc 3: Markdown Source
├── final_analysis_report.md         # Complete research & underwriter analysis report
└── README.md                        # Documentation Manual
```

---

## 🚀 How to Run & Deploy the Project

### 1. Run Locally
```bash
# Clone repository
git clone https://github.com/harshal25-hub/Loan-Amount-Prediction-ML.git
cd Loan-Amount-Prediction-ML

# Install requirements
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Deploy to Streamlit Community Cloud (Free Cloud Hosting)
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account (`harshal25-hub`).
2. Click **"New app"**.
3. Select Repository: `harshal25-hub/Loan-Amount-Prediction-ML`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **"Deploy!"**.
Your application will be live globally with a public `https://<your-app-name>.streamlit.app` URL!

---

## 📑 3 Comprehensive Educational Documents (with PDFs)

This repository includes 3 in-depth educational documents explaining **HOW, WHY, and WHEN** in simple English:

1. **[Doc 1: ML Topics Explanation](ml_topics_explanation.md)** ([Download PDF](ml_topics_explanation.pdf)):
   - Supervised Learning, Regression vs Classification
   - Feature Scaling, One-Hot Encoding, Missing Value Imputation
   - Train-Test Split, Bias-Variance Tradeoff
   - Evaluation Metrics (MAE, MSE, RMSE, R²)
   - Pipelines, Model Serialization & Streamlit Deployment

2. **[Doc 2: Project Files Explanation](project_files_explanation.md)** ([Download PDF](project_files_explanation.pdf)):
   - Detailed walkthrough of every single file in the codebase.
   - Code structure, inputs, outputs, and maintenance guidelines.

3. **[Doc 3: ML Models Explanation](ml_models_explanation.md)** ([Download PDF](ml_models_explanation.pdf)):
   - Mathematical and intuitive guide to the 4 regression algorithms.
   - Pros, cons, production guidance, and comparative performance rankings.
