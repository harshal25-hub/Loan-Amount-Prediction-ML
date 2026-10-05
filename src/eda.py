import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, sans-serif'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

def perform_eda(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    print("="*60)
    print("LOAN AMOUNT PREDICTION - EXPLORATORY DATA ANALYSIS (REAL DATA)")
    print("="*60)
    print(f"Total Records: {len(df)}")
    print(f"Features: {list(df.columns)}")
    print("\nData Types and Missing Values:")
    print(df.isnull().sum())
    
    # 1. Target Variable Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['LoanAmount'] / 1e5, kde=True, ax=axes[0], color='#2563eb', bins=35)
    axes[0].set_title("Distribution of Sanctioned Loan Amount (INR Lakhs)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Loan Amount (INR Lakhs)", fontsize=11)
    axes[0].set_ylabel("Applicant Count", fontsize=11)
    
    sns.histplot(np.log1p(df['LoanAmount']), kde=True, ax=axes[1], color='#059669', bins=35)
    axes[1].set_title("Log-Transformed Loan Amount Distribution", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Log(Loan Amount)", fontsize=11)
    axes[1].set_ylabel("Applicant Count", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loan_amount_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Correlation Matrix Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # 3. Requested vs Sanctioned Loan Amount
    plt.figure(figsize=(9, 6))
    sns.regplot(data=df, x=df['LoanAmountRequest']/1e5, y=df['LoanAmount']/1e5,
                scatter_kws={'alpha': 0.35, 'color': '#3b82f6', 's': 20},
                line_kws={'color': '#ef4444', 'linewidth': 2.5})
    plt.title("Loan Amount Requested vs Sanctioned (INR Lakhs)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Requested Loan Amount (INR Lakhs)", fontsize=11)
    plt.ylabel("Sanctioned Loan Amount (INR Lakhs)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'request_vs_sanctioned.png'), dpi=300)
    plt.close()
    
    # 4. Credit Score Impact (Binned Credit Score vs Loan Amount)
    df_temp = df.copy()
    df_temp['Credit_Tier'] = pd.cut(
        df_temp['CreditScore'],
        bins=[299, 600, 680, 750, 900],
        labels=['Poor (<600)', 'Fair (600-680)', 'Good (680-750)', 'Excellent (750+)']
    )
    
    plt.figure(figsize=(9, 6))
    sns.boxplot(x='Credit_Tier', y=df_temp['LoanAmount']/1e5, data=df_temp, palette='Blues_r', hue='Credit_Tier', legend=False)
    plt.title("Loan Amount Distribution Across Credit Score Tiers", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Credit Score Tier", fontsize=11)
    plt.ylabel("Loan Amount (INR Lakhs)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'credit_score_impact.png'), dpi=300)
    plt.close()
    
    # 5. Income & Asset Impact
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=df, x=df['ApplicantIncome']/1e5, y=df['LoanAmount']/1e5,
                    hue='IncomeStability', palette='Set1', alpha=0.5, ax=axes[0], s=25)
    axes[0].set_title("Applicant Income vs Loan Amount", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Monthly Income (INR Lakhs)", fontsize=11)
    axes[0].set_ylabel("Sanctioned Loan (INR Lakhs)", fontsize=11)
    
    sns.scatterplot(data=df, x=df['AssetValue']/1e5, y=df['LoanAmount']/1e5,
                    color='#8b5cf6', alpha=0.5, ax=axes[1], s=25)
    axes[1].set_title("Collateral Asset Value vs Loan Amount", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Property Asset Value (INR Lakhs)", fontsize=11)
    axes[1].set_ylabel("Sanctioned Loan (INR Lakhs)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'income_asset_impact.png'), dpi=300)
    plt.close()
    
    # 6. Location & Defaults Impact
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x='Location', y=df['LoanAmount']/1e5, hue='Location', data=df, ax=axes[0], palette='viridis', errorbar=None, legend=False)
    axes[0].set_title("Average Loan Amount by Location", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Location", fontsize=11)
    axes[0].set_ylabel("Avg Loan (INR Lakhs)", fontsize=11)
    
    sns.barplot(x='DefaultsCount', y=df['LoanAmount']/1e5, hue='DefaultsCount', data=df, ax=axes[1], palette='magma', errorbar=None, legend=False)
    axes[1].set_title("Impact of Past Defaults on Loan Amount", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Number of Defaults", fontsize=11)
    axes[1].set_ylabel("Avg Loan (INR Lakhs)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'location_defaults_impact.png'), dpi=300)
    plt.close()
    
    print("\nEDA visualizations generated and saved to:", output_dir)
    print("\nTop Correlations with Target (LoanAmount):")
    print(corr['LoanAmount'].sort_values(ascending=False))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, 'data', 'loan_data.csv')
    plots_dir = os.path.join(base_dir, 'plots')
    perform_eda(csv_file, plots_dir)
