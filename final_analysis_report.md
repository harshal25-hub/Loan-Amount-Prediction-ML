# Loan Amount Prediction Using Machine Learning
## Comprehensive Underwriting & Model Comparative Analysis Report

---

### 1. Executive Summary
This report presents an end-to-end Machine Learning system developed to predict sanctioned loan amounts for prospective applicants in a retail financial institution. Utilizing applicant demographics, income structures, employment stability, credit profiles (CIBIL), existing debt obligations, and collateral asset values, we trained, evaluated, and compared four distinct regression algorithms on a real-world online dataset (10,000 clean records).

The best-performing model, **Gradient Boosting Regression**, achieved an **$R^2$ score of 0.9888** and a **Root Mean Squared Error (RMSE) of ₹3,91,191.33** (a mean relative error of under 4.7%), outperforming Linear Regression, Decision Tree Regression, and Random Forest Regression. The production pipeline has been deployed as an interactive, real-time Streamlit underwriting application.

---

### 2. Problem Statement
A modern retail lending institution must estimate the optimal credit limit and loan sanction amount for applicants without exceeding safe debt-service limits or exposing the balance sheet to elevated default hazards. 

Manual loan underwriting suffers from:
1. **Processing Latency:** Days of multi-tier human credit review.
2. **Subjectivity & Human Inconsistency:** Inconsistent loan amounts granted to borrowers with similar financial capabilities.
3. **Complex Non-Linear Risk Dynamics:** Fixed Obligation to Income Ratios (FOIR), non-linear present value discounting curves of interest, and credit score cliff effects are difficult for manual spreadsheets to evaluate cohesively.

**Objective:** Develop a production-grade regression ML system that accurately predicts the expected sanctioned loan amount in Indian Rupees (₹) based on verified applicant parameters.

---

### 3. Objectives Achieved
- **Analyzed Real Loan Application Data:** Explored an authentic online dataset of 10,000 multi-attribute loan applications with zero data leakage.
- **Identified Critical Predictors:** Quantified feature correlations, debt burdens, and demographic elasticity.
- **Performed Rigorous Preprocessing & EDA:** Implemented standard scaling on continuous features, one-hot encoding on categorical attributes, log-transformed target analysis, and generated 10 high-resolution analytical visual plots.
- **Developed 4 Core Regression Models:**
  1. Linear Regression
  2. Decision Tree Regression
  3. Random Forest Regression
  4. Gradient Boosting Regression
- **Comparative Study:** Formally compared MAE, MSE, RMSE, and $R^2$ across all models on an unseen 20% holdout test set.
- **Deployment:** Deployed an interactive, modern Streamlit web application returning real-time predictions (`Predicted Loan Amount: ₹X`).

---

### 4. Machine Learning Algorithms & Mathematical Formulations

```
+----------------------------------------------------------------------------------------------------+
|                                    INPUT DATA PREPROCESSING PIPELINE                               |
|                                                                                                    |
|  Numerical: [ApplicantIncome, LoanAmountRequest, ExistingLiabilities, Dependents, CreditScore,      |
|              DefaultsCount, CoApplicant, AssetValue]  --->  StandardScaler()                       |
|                                                                                                    |
|  Categorical: [IncomeStability, Profession, EmploymentType, Location, ActiveCreditCard,             |
|                PropertyLocation]  ---> OneHotEncoder()                                             |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │              4 REGRESSION ALGORITHMS            │
                         └─────────────────────────────────────────────────┘
                                 │              │              │              │
                                 ▼              ▼              ▼              ▼
                           1. Linear Reg   2. Decision Tree   3. Random Forest  4. Gradient Boosting (Best)
                           R²: 0.9847      R²: 0.9861         R²: 0.9883        R²: 0.9888
                           RMSE: ₹4.56L    RMSE: ₹4.34L       RMSE: ₹3.99L      RMSE: ₹3.91L
```

#### 4.1 Linear Regression
- **Mechanism:** Models the target as a linear combination of features: $\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$.
- **Behavior:** Serves as the foundational baseline. Extremely fast and interpretable ($R^2 = 0.9847$, $RMSE = ₹4,56,024.85$).

#### 4.2 Decision Tree Regression
- **Mechanism:** Recursively partitions the feature space into orthogonal hyper-rectangles using Mean Squared Error criterion ($max\_depth=9$, $min\_samples\_leaf=10$).
- **Behavior:** Captures segmented credit tiers and policy rules, reducing RMSE to ₹4,34,535.01 ($R^2 = 0.9861$).

#### 4.3 Random Forest Regression
- **Mechanism:** Ensemble of 100 de-correlated bootstrap trees ($max\_depth=12$). Aggregates predictions via bootstrap bagging: $\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$.
- **Behavior:** Drastically lowers prediction variance compared to single trees, achieving an $R^2$ of 0.9883 and RMSE of ₹3,99,623.06.

#### 4.4 Gradient Boosting Regression (Optimal Champion)
- **Mechanism:** Builds additive regression trees sequentially:
  $$F_m(x) = F_{m-1}(x) + \eta \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$$
  Each step fits a new shallow tree to the pseudo-residuals with learning rate shrinkage ($\eta = 0.08$).
- **Behavior:** Struck the optimal bias-variance balance ($R^2 = 0.9888$, $RMSE = ₹3,91,191.33$, $MAE = ₹2,50,815.84$), capturing subtle non-linearities with maximum empirical precision.

---

### 5. Comparative Study

Evaluated on the unseen test set ($N = 2,000$ applications, 20% holdout):

| Rank | Algorithm | MAE (₹) | MSE (₹²) | RMSE (₹) | Test $R^2$ | Train $R^2$ | Overfitting Ratio | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Gradient Boosting Regression** | **₹2,50,815.84** | **1.53 × 10¹¹** | **₹3,91,191.33** | **0.9888** | 0.9948 | 1.006 | **Optimal Champion** |
| **#2** | **Random Forest Regression** | ₹2,58,950.02 | 1.60 × 10¹¹ | ₹3,99,623.06 | 0.9883 | 0.9938 | 1.005 | Robust Ensemble |
| **#3** | **Decision Tree Regression** | ₹2,74,960.44 | 1.89 × 10¹¹ | ₹4,34,535.01 | 0.9861 | 0.9900 | 1.004 | Rule Partitioning |
| **#4** | **Linear Regression** | ₹3,04,394.08 | 2.08 × 10¹¹ | ₹4,56,024.85 | 0.9847 | 0.9864 | 1.001 | Fast Baseline |

---

### 6. Deployment & User Interface

The system is deployed using an interactive Streamlit application:
- **Application URL:** `http://localhost:8501`.
- **Primary User Interaction:** Users specify applicant financial parameters and receive the instant sanction output:
  $$\text{Predicted Loan Amount: ₹X}$$
  (e.g., `Predicted Loan Amount: ₹50,11,855 (Approx. 50.12 Lakh)`).
- **Underwriting Support Features:**
  - Dynamic Credit Score indicator (color-coded Prime, Good, Fair, Subprime).
  - FOIR (Fixed Obligation to Income Ratio) calculation.
  - Maximum monthly disposable repayment capacity.
  - Multi-model comparison table displayed simultaneously for every applicant.
  - Download buttons for comprehensive PDF project documentation.

---

### 7. Final Analysis

#### 7.1 Factors Influencing Loan Amount
1. **Requested Loan Amount ($r = 0.991$):** Forms the initial requested baseline for underwriting.
2. **Collateral Asset Valuation ($r = 0.956$):** Acts as collateral backing, enabling lenders to safely extend higher Loan-To-Value (LTV) ratios.
3. **Existing Liabilities ($r = 0.690$):** Ongoing monthly debt obligations act as a direct deduction on borrowing capacity.
4. **Credit Score (CIBIL):** Functions as a critical gatekeeper; sub-600 scores trigger immediate sanction haircuts.
5. **Applicant Income:** Determines base serviceability.

#### 7.2 Best Regression Model
**Gradient Boosting Regression** emerged as the superior model across all statistical dimensions:
- Lowest MAE: ₹2,50,815.84
- Lowest RMSE: ₹3,91,191.33
- Highest Coefficient of Determination: $R^2 = 0.9888$

#### 7.3 Prediction Error Analysis
- **Residual Distribution:** Residual errors ($y - \hat{y}$) follow a symmetric Gaussian distribution centered at zero with no directional bias.
- **Homoscedasticity:** Error variances remain stable across low-, mid-, and high-loan brackets.

#### 7.4 Model Generalization
- The model was validated using an 80/20 train/test split.
- The test $R^2$ of 0.9888 closely aligns with the training performance without overfitting divergence.

#### 7.5 Practical Limitations & Production Safeguards
1. **Unverified Income:** The model assumes declared income matches audited bank deposits / ITR.
2. **Interest Rate Cycles:** If central bank policy rates fluctuate, loan capacity factors shift. The model should undergo quarterly hyperparameter recalibration.
3. **Collateral Title Integrity:** Collateral asset values must be independently audited by certified valuation officers.

---

### 8. Questions Answered

#### 1. Can loan amount be predicted from applicant information?
**Yes.** Standard applicant financial parameters explain **98.88% of the variance** in sanctioned loan amounts ($R^2 = 0.9888$).

#### 2. Which factors have the greatest influence?
1. Requested Loan Amount (~72% relative importance)
2. Collateral Asset Valuation (~18% relative importance)
3. Existing Monthly Liabilities (~5% relative importance)
4. Credit Score / CIBIL Tier (~3% relative importance)
5. Applicant Income (~2% relative importance)

#### 3. Which regression algorithm performs best?
**Gradient Boosting Regression** performs best, demonstrating superior handling of non-linear financial interactions and credit tier cutoffs compared to the other algorithms.

#### 4. Which model has the lowest RMSE?
**Gradient Boosting Regression** achieved the lowest RMSE of **₹3,91,191.33**, outperforming Random Forest (₹3.99L), Decision Tree (₹4.34L), and Linear Regression (₹4.56L).

#### 5. How accurately can the model predict loan amounts for new applicants?
On completely unseen applications, the champion model achieves a **Mean Absolute Error (MAE) of ₹2.50 Lakhs** on an average sanctioned loan portfolio of ₹53.4 Lakhs—a relative error rate of **under 4.7%**.

#### 6. Can this model assist loan officers?
**Yes, decisively.** The system delivers instant pre-qualification (under 50ms), removes subjective underwriter bias, standardizes FOIR calculations, and establishes an objective, data-driven loan ceiling before credit committee approval.
