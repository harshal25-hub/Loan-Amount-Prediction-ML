import os
import pandas as pd
import numpy as np

def clean_and_prepare_online_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'real_loan_dataset.csv')
    clean_path = os.path.join(base_dir, 'data', 'loan_data.csv')
    
    print(f"Loading raw online dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"Original shape: {df.shape}")
    
    # 1. Filter valid positive target records (real sanctioned loan amounts)
    df = df[df['Loan Sanction Amount (USD)'] > 0].copy()
    
    # 2. Impute / Clean missing values
    # Income (USD)
    median_income = df['Income (USD)'].median()
    df['Income (USD)'] = df['Income (USD)'].fillna(median_income)
    
    # Income Stability
    df['Income Stability'] = df['Income Stability'].fillna('Low')
    
    # Type of Employment
    df['Type of Employment'] = df['Type of Employment'].fillna('Other')
    
    # Current Loan Expenses (USD)
    df['Current Loan Expenses (USD)'] = df['Current Loan Expenses (USD)'].fillna(df['Current Loan Expenses (USD)'].median())
    
    # Dependents
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].median()).clip(0, 5).astype(int)
    
    # Credit Score
    median_credit = df['Credit Score'].median()
    df['Credit Score'] = df['Credit Score'].fillna(median_credit).clip(300, 850).round(1)
    
    # Active Credit Card
    df['Has Active Credit Card'] = df['Has Active Credit Card'].fillna('Unpossessed')
    
    # Property Location
    df['Property Location'] = df['Property Location'].fillna(df['Location'])
    
    # Co-Applicant
    df['Co-Applicant'] = df['Co-Applicant'].fillna(0).astype(int)
    
    # Currency Conversion: Convert USD to INR (₹) at standard ₹80/USD rate
    usd_to_inr = 80.0
    
    clean_df = pd.DataFrame({
        'ApplicantIncome': (df['Income (USD)'] * usd_to_inr).round(-2).astype(int),
        'IncomeStability': df['Income Stability'].astype(str),
        'Profession': df['Profession'].astype(str),
        'EmploymentType': df['Type of Employment'].astype(str),
        'Location': df['Location'].astype(str),
        'LoanAmountRequest': (df['Loan Amount Request (USD)'] * usd_to_inr).round(-2).astype(int),
        'ExistingLiabilities': (df['Current Loan Expenses (USD)'] * usd_to_inr).round(-2).astype(int),
        'Dependents': df['Dependents'].astype(int),
        'CreditScore': df['Credit Score'].round().astype(int),
        'DefaultsCount': df['No. of Defaults'].fillna(0).astype(int),
        'ActiveCreditCard': df['Has Active Credit Card'].astype(str),
        'PropertyLocation': df['Property Location'].astype(str),
        'CoApplicant': df['Co-Applicant'].astype(int),
        'AssetValue': (df['Property Price'] * usd_to_inr).round(-2).astype(int),
        'LoanAmount': (df['Loan Sanction Amount (USD)'] * usd_to_inr).round(-2).astype(int)
    })
    
    # Clean anomalies (remove any row with AssetValue <= 0 or LoanAmount <= 0)
    clean_df = clean_df[(clean_df['AssetValue'] > 0) & (clean_df['LoanAmount'] > 0)]
    
    # Take a statistically robust sample (e.g. 10,000 records for fast training and rock-solid evaluation)
    if len(clean_df) > 10000:
        clean_df = clean_df.sample(n=10000, random_state=42).reset_index(drop=True)
        
    clean_df.to_csv(clean_path, index=False)
    print(f"Cleaned online dataset saved to: {clean_path}")
    print(f"Shape: {clean_df.shape}")
    print("\nSummary Statistics of Target (LoanAmount in INR):")
    print(clean_df['LoanAmount'].describe())
    print("\nSample records:")
    print(clean_df.head(2))

if __name__ == '__main__':
    clean_and_prepare_online_dataset()
