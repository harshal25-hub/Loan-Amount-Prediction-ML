import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from PIL import Image

from src.predict import predict_loan_amount, METRICS, METADATA, format_inr, format_inr_words

# Configure Page
st.set_page_config(
    page_title="CrediPredict ML — Loan Amount Prediction System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .hero-amount {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(180deg, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 5px 0;
    }
    .hero-subtitle {
        color: #38bdf8;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .pill-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .pill-prime { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .pill-good { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
    .pill-fair { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .pill-subprime { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("💼 CrediPredict ML — Loan Amount Prediction")
st.markdown("##### Intelligent Machine Learning Underwriting System trained on Real Online Banking Applications")

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Choose View:",
    ["💼 Loan Predictor", "📊 4-Model Benchmark", "📈 Exploratory Data Analysis", "📑 Executive Analysis & Q&A", "📥 Download PDF Documents"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Production Model")
st.sidebar.info("**Gradient Boosting Regressor**\n- **R² Score:** 0.9888\n- **RMSE:** ₹3,91,191\n- **Trained on:** 10,000 real records")

# Sample Profiles
PROFILES = {
    "Select a pre-filled profile...": None,
    "💻 Tech Salaried (₹3.1L/mo, Credit 780, High Asset)": {
        'ApplicantIncome': 310000,
        'IncomeStability': 'High',
        'Profession': 'Commercial associate',
        'EmploymentType': 'Core staff',
        'Location': 'Urban',
        'LoanAmountRequest': 4500000,
        'ExistingLiabilities': 25000,
        'Dependents': 1,
        'CreditScore': 780,
        'DefaultsCount': 0,
        'ActiveCreditCard': 'Active',
        'PropertyLocation': 'Urban',
        'CoApplicant': 1,
        'AssetValue': 8500000
    },
    "🏢 Business Owner (₹4.8L/mo, Credit 745)": {
        'ApplicantIncome': 480000,
        'IncomeStability': 'High',
        'Profession': 'Commercial associate',
        'EmploymentType': 'Managers',
        'Location': 'Semi-Urban',
        'LoanAmountRequest': 7500000,
        'ExistingLiabilities': 45000,
        'Dependents': 2,
        'CreditScore': 745,
        'DefaultsCount': 0,
        'ActiveCreditCard': 'Active',
        'PropertyLocation': 'Semi-Urban',
        'CoApplicant': 1,
        'AssetValue': 14000000
    },
    "🌾 Rural Borrower (₹1.1L/mo, Credit 660)": {
        'ApplicantIncome': 110000,
        'IncomeStability': 'Low',
        'Profession': 'Working',
        'EmploymentType': 'Laborers',
        'Location': 'Rural',
        'LoanAmountRequest': 2000000,
        'ExistingLiabilities': 8000,
        'Dependents': 2,
        'CreditScore': 660,
        'DefaultsCount': 0,
        'ActiveCreditCard': 'Unpossessed',
        'PropertyLocation': 'Rural',
        'CoApplicant': 0,
        'AssetValue': 3000000
    },
    "⚠️ High Debt / Low Score (₹1.8L/mo, Credit 570)": {
        'ApplicantIncome': 180000,
        'IncomeStability': 'Low',
        'Profession': 'Working',
        'EmploymentType': 'Drivers',
        'Location': 'Semi-Urban',
        'LoanAmountRequest': 3500000,
        'ExistingLiabilities': 70000,
        'Dependents': 3,
        'CreditScore': 570,
        'DefaultsCount': 1,
        'ActiveCreditCard': 'Inactive',
        'PropertyLocation': 'Semi-Urban',
        'CoApplicant': 1,
        'AssetValue': 2500000
    }
}

# TAB 1: PREDICTOR
if menu == "💼 Loan Predictor":
    st.subheader("Enter Applicant Information")
    
    selected_profile = st.selectbox("Quick Demo Profile Fillers:", list(PROFILES.keys()))
    default_vals = PROFILES.get(selected_profile) or {}

    col1, col2 = st.columns([1.1, 0.9])
    
    with col1:
        with st.form("loan_form"):
            st.markdown("##### 1. Income & Employment")
            c1, c2 = st.columns(2)
            with c1:
                applicant_income = st.number_input(
                    "Monthly Income (₹)", 
                    min_value=20000, max_value=2000000, 
                    value=int(default_vals.get('ApplicantIncome', 250000)), step=10000
                )
                income_stability = st.selectbox(
                    "Income Stability", 
                    ['Low', 'High'], 
                    index=0 if default_vals.get('IncomeStability', 'Low') == 'Low' else 1
                )
                profession = st.selectbox(
                    "Profession", 
                    ['Commercial associate', 'Working', 'Pensioner', 'State servant'],
                    index=0
                )
            with c2:
                employment_type = st.selectbox(
                    "Employment Type",
                    ['Sales staff', 'Managers', 'Laborers', 'Core staff', 'Drivers', 'Other'],
                    index=3
                )
                location = st.selectbox(
                    "Location", 
                    ['Urban', 'Semi-Urban', 'Rural'],
                    index=0
                )
                co_applicant = st.selectbox(
                    "Co-Applicant Available?", 
                    [1, 0], 
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    index=0 if default_vals.get('CoApplicant', 1) == 1 else 1
                )
                
            st.markdown("##### 2. Credit Profile & Existing Liabilities")
            c3, c4 = st.columns(2)
            with c3:
                credit_score = st.slider(
                    "Credit Score (CIBIL)", 
                    min_value=300, max_value=850, 
                    value=int(default_vals.get('CreditScore', 760))
                )
                existing_liabilities = st.number_input(
                    "Existing Monthly EMIs / Liabilities (₹)", 
                    min_value=0, max_value=500000, 
                    value=int(default_vals.get('ExistingLiabilities', 15000)), step=2000
                )
            with c4:
                defaults_count = st.number_input(
                    "Past Defaults Count", 
                    min_value=0, max_value=5, 
                    value=int(default_vals.get('DefaultsCount', 0))
                )
                active_card = st.selectbox(
                    "Active Credit Card", 
                    ['Active', 'Inactive', 'Unpossessed'],
                    index=0
                )
                
            st.markdown("##### 3. Loan Request & Collateral")
            c5, c6 = st.columns(2)
            with c5:
                loan_request = st.number_input(
                    "Loan Amount Requested (₹)", 
                    min_value=100000, max_value=40000000, 
                    value=int(default_vals.get('LoanAmountRequest', 4500000)), step=50000
                )
                dependents = st.number_input(
                    "Number of Dependents", 
                    min_value=0, max_value=6, 
                    value=int(default_vals.get('Dependents', 1))
                )
            with c6:
                asset_value = st.number_input(
                    "Collateral Property / Asset Value (₹)", 
                    min_value=200000, max_value=60000000, 
                    value=int(default_vals.get('AssetValue', 6500000)), step=100000
                )
                property_location = st.selectbox(
                    "Property Location", 
                    ['Urban', 'Semi-Urban', 'Rural'],
                    index=0
                )

            st.markdown("##### 4. Model Selection")
            model_selected = st.selectbox(
                "Choose Regression Algorithm for Estimation:",
                [
                    "Gradient Boosting Regression",
                    "Random Forest Regression",
                    "Decision Tree Regression",
                    "Linear Regression"
                ]
            )

            submitted = st.form_submit_button("⚡ Calculate Sanctioned Loan Amount", use_container_width=True)

    with col2:
        applicant_data = {
            'ApplicantIncome': applicant_income,
            'IncomeStability': income_stability,
            'Profession': profession,
            'EmploymentType': employment_type,
            'Location': location,
            'LoanAmountRequest': loan_request,
            'ExistingLiabilities': existing_liabilities,
            'Dependents': dependents,
            'CreditScore': credit_score,
            'DefaultsCount': defaults_count,
            'ActiveCreditCard': active_card,
            'PropertyLocation': property_location,
            'CoApplicant': co_applicant,
            'AssetValue': asset_value
        }
        
        result = predict_loan_amount(applicant_data, model_name=model_selected)
        
        st.markdown(f"""
        <div class="metric-card">
            <span class="pill-badge pill-{result['risk_badge']}">{result['risk_category']}</span>
            <div style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;">Estimated Loan Sanction Limit</div>
            <div class="hero-amount">{result['formatted_loan_amount']}</div>
            <div class="hero-subtitle">Approx. {result['formatted_loan_words']}</div>
            <hr style="border-color: rgba(255,255,255,0.08); margin: 12px 0;">
            <div style="font-size: 0.85rem; color: #94a3b8;">
                Evaluated by: <strong style="color: #ffffff;">{result['model_used']}</strong> 
                (Test R²: <strong>{result['model_metrics'].get('R2', 0):.4f}</strong>)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Monthly Income", format_inr(applicant_income))
        with m2:
            st.metric("Max EMI Capacity", result['max_monthly_emi_capacity'])
        with m3:
            st.metric("Current FOIR", result['current_foir_percent'])

        st.markdown("#### 🔍 Algorithmic Comparison for this Applicant")
        comp_rows = []
        for m_name, info in result['all_model_predictions'].items():
            comp_rows.append({
                "Algorithm": m_name,
                "Predicted Sanction": info['formatted'],
                "Approx (Words)": info['formatted_words'],
                "Model RMSE": format_inr(info['rmse']),
                "Model R²": f"{info['r2']:.4f}"
            })
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

# TAB 2: 4-MODEL BENCHMARK
elif menu == "📊 4-Model Benchmark":
    st.subheader("4-Model Comparative Study & Benchmark")
    st.markdown("All 4 regression models were trained with standard preprocessing and evaluated on an unseen 20% holdout test set (2,000 real records):")
    
    benchmark_data = [
        {"Rank": "#1", "Model": "Gradient Boosting Regression", "MAE": "₹2,50,815.84", "MSE": "1.53e+11", "RMSE": "₹3,91,191.33", "Test R²": 0.9888, "Train R²": 0.9948, "Status": "Optimal (Best)"},
        {"Rank": "#2", "Model": "Random Forest Regression", "MAE": "₹2,58,950.02", "MSE": "1.60e+11", "RMSE": "₹3,99,623.06", "Test R²": 0.9883, "Train R²": 0.9938, "Status": "Robust Ensemble"},
        {"Rank": "#3", "Model": "Decision Tree Regression", "MAE": "₹2,74,960.44", "MSE": "1.89e+11", "RMSE": "₹4,34,535.01", "Test R²": 0.9861, "Train R²": 0.9900, "Status": "Step Non-Linearity"},
        {"Rank": "#4", "Model": "Linear Regression", "MAE": "₹3,04,394.08", "MSE": "2.08e+11", "RMSE": "₹4,56,024.85", "Test R²": 0.9847, "Train R²": 0.9864, "Status": "Fast Baseline"}
    ]
    st.table(pd.DataFrame(benchmark_data))

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("plots/model_comparison_metrics.png"):
            st.image("plots/model_comparison_metrics.png", caption="Model Comparison (RMSE, MAE & R²)", use_container_width=True)
        if os.path.exists("plots/residuals_distribution.png"):
            st.image("plots/residuals_distribution.png", caption="Residual Error Distribution", use_container_width=True)
    with c2:
        if os.path.exists("plots/actual_vs_predicted.png"):
            st.image("plots/actual_vs_predicted.png", caption="Actual vs Predicted (Gradient Boosting)", use_container_width=True)
        if os.path.exists("plots/feature_importance.png"):
            st.image("plots/feature_importance.png", caption="Top 10 Feature Importances", use_container_width=True)

# TAB 3: EDA
elif menu == "📈 Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis (EDA) on Real Loan Dataset")
    
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("plots/correlation_heatmap.png"):
            st.image("plots/correlation_heatmap.png", caption="Feature Correlation Heatmap", use_container_width=True)
        if os.path.exists("plots/request_vs_sanctioned.png"):
            st.image("plots/request_vs_sanctioned.png", caption="Requested vs Sanctioned Amount", use_container_width=True)
        if os.path.exists("plots/income_asset_impact.png"):
            st.image("plots/income_asset_impact.png", caption="Income & Asset Value Impact", use_container_width=True)
    with c2:
        if os.path.exists("plots/loan_amount_distribution.png"):
            st.image("plots/loan_amount_distribution.png", caption="Target Variable Distribution", use_container_width=True)
        if os.path.exists("plots/credit_score_impact.png"):
            st.image("plots/credit_score_impact.png", caption="Credit Score Tiers vs Loan Amount", use_container_width=True)
        if os.path.exists("plots/location_defaults_impact.png"):
            st.image("plots/location_defaults_impact.png", caption="Location and Defaults Breakdown", use_container_width=True)

# TAB 4: EXECUTIVE REPORT & Q&A
elif menu == "📑 Executive Analysis & Q&A":
    st.subheader("Answers to the 6 Core Questions (Section 8)")
    
    questions = [
        ("1. Can loan amount be predicted from applicant information?",
         "**Yes, with exceptional empirical precision.** Applicant income, requested loan amount, credit score, collateral asset valuation, and liabilities explain **98.88% of the variance** in sanctioned loan amounts ($R^2 = 0.9888$) using Gradient Boosting Regression."),
        
        ("2. Which factors have the greatest influence?",
         "1. **Requested Loan Amount ($r = 0.99$):** Primary baseline indicator of loan sizing.\n2. **Collateral Asset Value ($r = 0.96$):** Provides recovery security and fixes maximum LTV.\n3. **Existing Liabilities ($r = 0.69$):** Direct cash flow deduction on monthly debt capacity.\n4. **Credit Score (CIBIL):** Determines risk category and sanction haircuts.\n5. **Applicant Income:** Determines base serviceability."),
        
        ("3. Which regression algorithm performs best?",
         "**Gradient Boosting Regression** achieved the highest test $R^2$ (0.9888) and lowest RMSE (₹3.91 Lakhs). It sequentially fits residual errors of previous trees, capturing complex financial interactions and credit tier cutoffs."),
        
        ("4. Which model has the lowest RMSE?",
         "**Gradient Boosting Regression** with **RMSE = ₹3,91,191.33**, outperforming Random Forest (₹3.99L), Decision Tree (₹4.34L), and Linear Regression (₹4.56L)."),
        
        ("5. How accurately can the model predict loan amounts for new applicants?",
         "On completely unseen test applications (2,000 holdout records), the model exhibits an **MAE of ₹2.50 Lakhs** on loans averaging ₹53.4 Lakhs—a relative error rate of **under 4.7%**."),
        
        ("6. Can this model assist loan officers?",
         "**Yes.** It provides instant pre-qualification, eliminates manual underwriting subjectivity, ensures regulatory FOIR adherence, and computes objective credit sanction recommendations in sub-second time.")
    ]
    
    for q, a in questions:
        with st.expander(f"📌 {q}", expanded=True):
            st.markdown(a)

    st.markdown("---")
    st.subheader("Final Project Analysis (Section 7)")
    st.markdown("""
    - **Factors Influencing Loan Amount:** Retail credit underwriting is driven by serviceability (income & ongoing EMIs) and loss-given-default collateral (property asset price).
    - **Comparative Algorithm Insights:** Linear regression achieves a strong baseline ($R^2 = 0.9847$), while ensemble tree models capture edge-case thresholds and step policies with higher accuracy ($R^2 = 0.9888$).
    - **Prediction Error Analysis:** Errors follow a zero-mean Gaussian distribution with no directional bias.
    - **Practical Limitations:** Self-employed unverified cash flows, macroeconomic interest rate shifts, and legal title encumbrance require standard credit committee validation.
    """)

# TAB 5: DOWNLOAD PDF DOCS
elif menu == "📥 Download PDF Documents":
    st.subheader("Download Comprehensive Project Documentation (PDF)")
    st.markdown("All 3 detailed educational and technical explanation documents are available for download:")
    
    pdf_files = [
        ("1. ML Topics Explained (How, Why, When in Simple English)", "ml_topics_explanation.pdf"),
        ("2. Project Files Explained (How, Why, When in Simple English)", "project_files_explanation.pdf"),
        ("3. ML Models Explained (How, Why, When in Simple English)", "ml_models_explanation.pdf")
    ]
    
    for title, filename in pdf_files:
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label=f"📄 Download {title}",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning(f"File {filename} is generating...")
