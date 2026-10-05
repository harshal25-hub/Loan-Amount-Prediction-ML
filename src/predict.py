import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r') as f:
    METADATA = json.load(f)

with open(os.path.join(MODELS_DIR, 'metrics_comparison.json'), 'r') as f:
    METRICS = json.load(f)

LOADED_MODELS = {}

MODEL_FILES = {
    'Gradient Boosting Regression': 'gradient_boosting_regression.joblib',
    'Random Forest Regression': 'random_forest_regression.joblib',
    'Decision Tree Regression': 'decision_tree_regression.joblib',
    'Linear Regression': 'linear_regression.joblib'
}

def get_model(model_name='Gradient Boosting Regression'):
    if model_name not in LOADED_MODELS:
        filename = MODEL_FILES.get(model_name, 'best_model.joblib')
        path = os.path.join(MODELS_DIR, filename)
        LOADED_MODELS[model_name] = joblib.load(path)
    return LOADED_MODELS[model_name]

def format_inr(number):
    try:
        val = int(round(number))
        is_negative = val < 0
        val = abs(val)
        s = str(val)
        if len(s) <= 3:
            res = s
        else:
            last3 = s[-3:]
            rest = s[:-3]
            chunks = []
            while len(rest) > 2:
                chunks.insert(0, rest[-2:])
                rest = rest[:-2]
            if rest:
                chunks.insert(0, rest)
            res = ",".join(chunks) + "," + last3
        prefix = "-₹" if is_negative else "₹"
        return f"{prefix}{res}"
    except Exception:
        return f"₹{number:,.0f}"

def format_inr_words(number):
    val = float(number)
    if val >= 1e7:
        return f"{val / 1e7:.2f} Crore"
    elif val >= 1e5:
        return f"{val / 1e5:.2f} Lakh"
    elif val >= 1e3:
        return f"{val / 1e3:.1f} Thousand"
    return f"{val:.0f}"

def predict_loan_amount(applicant_dict, model_name='Gradient Boosting Regression'):
    model = get_model(model_name)
    df_input = pd.DataFrame([applicant_dict])
    
    predicted_val = float(model.predict(df_input)[0])
    predicted_val = max(50000.0, predicted_val)
    
    all_predictions = {}
    for name in MODEL_FILES.keys():
        m = get_model(name)
        val = max(50000.0, float(m.predict(df_input)[0]))
        all_predictions[name] = {
            'predicted_amount': round(val),
            'formatted': format_inr(val),
            'formatted_words': format_inr_words(val),
            'rmse': METRICS[name]['RMSE'],
            'r2': METRICS[name]['R2']
        }
        
    income = float(applicant_dict.get('ApplicantIncome', 100000))
    liabilities = float(applicant_dict.get('ExistingLiabilities', 10000))
    credit_score = int(applicant_dict.get('CreditScore', 750))
    
    foir = (liabilities / income * 100) if income > 0 else 0
    max_emi = max(0, (income * 0.50) - liabilities)
    
    if credit_score >= 750:
        risk_category = "Prime / Excellent (Fast-Track Approval)"
        risk_badge = "success"
    elif credit_score >= 680:
        risk_category = "Good / Low Risk"
        risk_badge = "primary"
    elif credit_score >= 600:
        risk_category = "Fair / Moderate Risk"
        risk_badge = "warning"
    else:
        risk_category = "Subprime / High Risk"
        risk_badge = "danger"

    return {
        'model_used': model_name,
        'predicted_amount': round(predicted_val),
        'formatted_loan_amount': format_inr(predicted_val),
        'formatted_loan_words': format_inr_words(predicted_val),
        'max_monthly_emi_capacity': format_inr(max_emi),
        'current_foir_percent': f"{foir:.1f}%",
        'risk_category': risk_category,
        'risk_badge': risk_badge,
        'all_model_predictions': all_predictions,
        'model_metrics': METRICS.get(model_name, {})
    }
