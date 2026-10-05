import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import joblib

def build_and_evaluate_models(data_path, output_dir, plots_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    print(f"Loaded real dataset from {data_path} with shape: {df.shape}")
    
    X = df.drop(columns=['LoanAmount'])
    y = df['LoanAmount']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    numerical_features = [
        'ApplicantIncome', 'LoanAmountRequest', 'ExistingLiabilities',
        'Dependents', 'CreditScore', 'DefaultsCount', 'CoApplicant', 'AssetValue'
    ]
    
    categorical_features = [
        'IncomeStability', 'Profession', 'EmploymentType',
        'Location', 'ActiveCreditCard', 'PropertyLocation'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 6 Algorithms:
    # 1. Linear Regression
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # 2. Polynomial Regression (Degree 2 with Ridge regularizer)
    poly_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', Ridge(alpha=100.0))
    ])
    
    # 3. Decision Tree Regression
    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=9, min_samples_leaf=10, random_state=42))
    ])
    
    # 4. Random Forest Regression
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1))
    ])
    
    # 5. Gradient Boosting Regression
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42))
    ])
    
    # 6. Support Vector Regression (SVR with target scaling)
    svr_base = Pipeline([
        ('preprocessor', preprocessor),
        ('svr', SVR(kernel='rbf', C=30.0, epsilon=0.05))
    ])
    svr_pipeline = TransformedTargetRegressor(
        regressor=svr_base,
        transformer=StandardScaler()
    )
    
    models = {
        'Linear Regression': lr_pipeline,
        'Polynomial Regression': poly_pipeline,
        'Decision Tree Regression': dt_pipeline,
        'Random Forest Regression': rf_pipeline,
        'Gradient Boosting Regression': gb_pipeline,
        'Support Vector Regression': svr_pipeline
    }
    
    results = {}
    fitted_models = {}
    test_predictions = {}
    
    print("\n" + "="*80)
    print("TRAINING AND EVALUATING 6 REGRESSION MODELS (REAL ONLINE DATASET)")
    print("="*80)
    
    for name, model in models.items():
        print(f"\nTraining [{name}]...")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        test_predictions[name] = y_test_pred
        
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        results[name] = {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'R2': float(r2),
            'Train_RMSE': float(train_rmse),
            'Train_R2': float(train_r2)
        }
        
        print(f"Results for {name}:")
        print(f"  MAE  : ₹{mae:,.2f}")
        print(f"  MSE  : {mse:,.2e}")
        print(f"  RMSE : ₹{rmse:,.2f}")
        print(f"  R²   : {r2:.4f} (Train R²: {train_r2:.4f})")
        
        clean_name = name.lower().replace(' ', '_')
        joblib.dump(model, os.path.join(output_dir, f"{clean_name}.joblib"))
        
    ranked_models = sorted(results.items(), key=lambda x: x[1]['RMSE'])
    best_model_name = ranked_models[0][0]
    best_model_metrics = ranked_models[0][1]
    
    print("\n" + "="*80)
    print(f"BEST REGRESSION MODEL: {best_model_name}")
    print(f"Lowest RMSE: ₹{best_model_metrics['RMSE']:,.2f} | Highest R²: {best_model_metrics['R2']:.4f}")
    print("="*80)
    
    joblib.dump(fitted_models[best_model_name], os.path.join(output_dir, "best_model.joblib"))
    
    metadata = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'categorical_options': {
            'IncomeStability': ['Low', 'High'],
            'Profession': ['Commercial associate', 'Working', 'Pensioner', 'State servant'],
            'EmploymentType': ['Sales staff', 'Managers', 'Laborers', 'Core staff', 'Drivers', 'Other'],
            'Location': ['Semi-Urban', 'Urban', 'Rural'],
            'ActiveCreditCard': ['Active', 'Inactive', 'Unpossessed'],
            'PropertyLocation': ['Semi-Urban', 'Urban', 'Rural']
        },
        'best_model': best_model_name,
        'models_compared': list(models.keys()),
        'test_records_count': len(y_test),
        'train_records_count': len(y_train)
    }
    
    with open(os.path.join(output_dir, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    with open(os.path.join(output_dir, "metrics_comparison.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    generate_comparison_plots(results, y_test, test_predictions, best_model_name, fitted_models, plots_dir)
    return results, best_model_name

def generate_comparison_plots(results, y_test, test_predictions, best_model_name, fitted_models, plots_dir):
    sns.set_theme(style="whitegrid")
    
    model_names = list(results.keys())
    rmse_vals = [results[m]['RMSE'] / 1e5 for m in model_names]
    mae_vals = [results[m]['MAE'] / 1e5 for m in model_names]
    r2_vals = [results[m]['R2'] for m in model_names]
    
    # 1. Comparative Metrics Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x - width/2, rmse_vals, width, label='RMSE (INR Lakhs)', color='#ef4444', alpha=0.85)
    axes[0].bar(x + width/2, mae_vals, width, label='MAE (INR Lakhs)', color='#3b82f6', alpha=0.85)
    axes[0].set_ylabel('Error in INR Lakhs', fontsize=12)
    axes[0].set_title('Error Comparison (RMSE vs MAE) Across Models', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace(' Regression', '') for m in model_names], rotation=25, ha='right', fontsize=10)
    axes[0].legend(fontsize=11)
    
    colors = ['#10b981' if m == best_model_name else '#64748b' for m in model_names]
    bars = axes[1].bar(x, r2_vals, width=0.5, color=colors)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('R² Coefficient of Determination by Model', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace(' Regression', '') for m in model_names], rotation=25, ha='right', fontsize=10)
    axes[1].set_ylim([0.8, 1.02])
    
    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold')
                         
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison_metrics.png'), dpi=300)
    plt.close()
    
    # 2. Actual vs Predicted for Best Model
    best_preds = test_predictions[best_model_name]
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test / 1e5, best_preds / 1e5, alpha=0.3, color='#2563eb', edgecolors='none', s=20)
    min_val = min(y_test.min(), best_preds.min()) / 1e5
    max_val = max(y_test.max(), best_preds.max()) / 1e5
    plt.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2.5, label='Perfect Prediction (y = ŷ)')
    
    plt.title(f"Actual vs Predicted Loan Amount ({best_model_name})", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Actual Loan Amount (INR Lakhs)", fontsize=11)
    plt.ylabel("Predicted Loan Amount (INR Lakhs)", fontsize=11)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    # 3. Residual Distribution for Best Model
    residuals = (y_test - best_preds) / 1e5
    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, kde=True, color='#8b5cf6', bins=40)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.title(f"Residual Error Distribution for {best_model_name}", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Residual Error (INR Lakhs)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_distribution.png'), dpi=300)
    plt.close()
    
    # 4. Feature Importance
    gb_model = fitted_models.get('Gradient Boosting Regression')
    if gb_model:
        preprocessor = gb_model.named_steps['preprocessor']
        cat_encoder = preprocessor.named_transformers_['cat']
        encoded_cat_names = list(cat_encoder.get_feature_names_out(['IncomeStability', 'Profession', 'EmploymentType', 'Location', 'ActiveCreditCard', 'PropertyLocation']))
        num_names = ['ApplicantIncome', 'LoanAmountRequest', 'ExistingLiabilities', 'Dependents', 'CreditScore', 'DefaultsCount', 'CoApplicant', 'AssetValue']
        all_features = num_names + encoded_cat_names
        
        importances = gb_model.named_steps['regressor'].feature_importances_
        feat_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_df.head(10), x='Importance', y='Feature', hue='Feature', palette='crest', legend=False)
        plt.title("Top 10 Feature Importances (Gradient Boosting)", fontsize=14, fontweight='bold', pad=12)
        plt.xlabel("Relative Importance Score", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'loan_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')
    build_and_evaluate_models(data_file, models_dir, plots_dir)
