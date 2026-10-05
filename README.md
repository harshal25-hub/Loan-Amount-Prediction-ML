# Loan Amount Prediction Using Machine Learning

An end-to-end Machine Learning system and interactive web application developed to estimate the appropriate loan sanction amount for applicants based on financial profiles, credit history, existing liabilities, and collateral backing.

---

## 🌟 Key Features
- **Exploratory Data Analysis (EDA):** In-depth statistical analysis and visualizations (correlations, credit tiers, distributions).
- **6 Regression Algorithms Implemented & Compared:**
  1. Linear Regression
  2. Polynomial Regression (Degree 2)
  3. Decision Tree Regression
  4. Random Forest Regression
  5. Gradient Boosting Regression *(Champion Model)*
  6. Support Vector Regression (SVR)
- **Comparative Study:** Complete benchmark table evaluating MAE, MSE, RMSE, and R² Score.
- **Interactive Web Application & Deployment:**
  - Enter applicant information and receive **`Predicted Loan Amount: ₹X`**.
  - Dynamic Credit Score slider with live CIBIL rating pill.
  - Live FOIR (Fixed Obligation to Income Ratio) & Max Monthly EMI Capacity calculator.
  - Multi-model comparative view for every individual applicant.
  - Pre-filled realistic demo applicant profiles.
- **REST API Endpoint:** `POST /api/predict` for programmatic integration.
- **Executive Analysis & Questions Answered:** Comprehensive report addressing all underwriting questions.

---

## 📊 Comparative Performance Summary

| Rank | Model | MAE (₹) | MSE | RMSE (₹) | Test R² | Train R² |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Gradient Boosting Regression** | **₹2,02,581** | **1.46e+11** | **₹3,82,446** | **0.9640** | **0.9957** |
| **#2** | Polynomial Regression (Deg-2) | ₹2,94,499 | 2.31e+11 | ₹4,81,090 | 0.9430 | 0.9505 |
| **#3** | Support Vector Regression (SVR) | ₹2,74,108 | 2.39e+11 | ₹4,89,285 | 0.9410 | 0.9951 |
| **#4** | Random Forest Regression | ₹2,67,707 | 2.49e+11 | ₹4,98,741 | 0.9387 | 0.9859 |
| **#5** | Decision Tree Regression | ₹4,47,363 | 5.47e+11 | ₹7,39,447 | 0.8653 | 0.9261 |
| **#6** | Linear Regression | ₹5,70,825 | 7.37e+11 | ₹8,58,677 | 0.8183 | 0.8095 |

---

## 📁 Project Structure

```
├── data/
│   ├── generate_dataset.py          # Synthetic realistic loan dataset generator
│   └── loan_data.csv                # 5,000 application records
├── models/
│   ├── best_model.joblib            # Champion Gradient Boosting Pipeline
│   ├── gradient_boosting_regression.joblib
│   ├── random_forest_regression.joblib
│   ├── polynomial_regression.joblib
│   ├── support_vector_regression.joblib
│   ├── decision_tree_regression.joblib
│   ├── linear_regression.joblib
│   ├── metrics_comparison.json      # Benchmark metrics
│   └── model_metadata.json          # Preprocessing & feature metadata
├── src/
│   ├── eda.py                       # Exploratory Data Analysis & visual plot generation
│   ├── train_models.py              # Pipelines, model training, evaluation & metrics
│   └── predict.py                   # Prediction engine, INR formatting & FOIR calculator
├── static/
│   ├── css/style.css                # Custom modern fintech dashboard stylesheet
│   ├── js/app.js                    # Interactive DOM & async API integration
│   └── images/plots/                # High-res evaluation & EDA charts
├── templates/
│   └── index.html                   # Dashboard UI
├── app.py                           # Flask web server & REST API
├── final_analysis_report.md         # Full research & analysis report
└── README.md
```

---

## 🚀 Getting Started

### 1. Requirements
Ensure Python 3.10+ is installed:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib flask
```

### 2. Run Exploratory Data Analysis
```bash
python3 src/eda.py
```

### 3. Train and Benchmark the 6 Models
```bash
python3 src/train_models.py
```

### 4. Launch the Web Application
```bash
python3 app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5001
```

---

## 🔌 REST API Usage

### Endpoint: `POST /api/predict`
**Request Payload:**
```json
{
  "ApplicantIncome": 85000,
  "CoapplicantIncome": 35000,
  "EmploymentType": "Salaried",
  "WorkExperience": 5.0,
  "Education": "Graduate",
  "CreditScore": 770,
  "ExistingLiabilities": 14000,
  "AssetValue": 5000000,
  "LoanTerm": 240,
  "PropertyArea": "Urban",
  "Dependents": 1,
  "MaritalStatus": "Married",
  "model_name": "Gradient Boosting Regression"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "formatted_loan_amount": "₹50,11,855",
    "formatted_loan_words": "50.12 Lakh",
    "predicted_amount": 5011855,
    "model_used": "Gradient Boosting Regression",
    "risk_category": "Prime / Excellent (Fast-Track Approval)",
    "current_foir_percent": "11.7%",
    "max_monthly_emi_capacity": "₹46,000",
    "formatted_total_income": "₹1,20,000",
    "all_model_predictions": { ... }
  }
}
```
