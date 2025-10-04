import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats



# DAY 1: DATA LOADING & BASIC INSPECTION


# print(" DAY 1: DATA LOADING & BASIC INSPECTION ")

# Load data
df = pd.read_csv('./lending_club_cleaned_with_target.csv')

# print(f"Portfolio: {len(df):,} loans | Default Rate: {df['default'].mean():.2%}")


# # DAY 2: DATA QUALITY ASSESSMENT

# print("\nDAY 2: DATA QUALITY ASSESSMENT ")

# Check for missing values
# print("Missing Values by Column:")
# missing_values = df.isnull().sum()
# print(missing_values[missing_values > 0])

# # Calculate missing percentage
# missing_percent = df.isnull().mean().sort_values(ascending=False)
# print("\nMissing Values Percentage (Top 10):")
# print(missing_percent.head(10))

# # DAY 3: UNIVARIATE ANALYSIS


# print("\nDAY  3: UNIVARIATE ANALYSIS")

# # 1. Portfolio Concentration Risk
# print("1. PORTFOLIO CONCENTRATION:")
# loan_amount_stats = df['loan_amnt'].describe()
# print(f"   • Max loan: ${loan_amount_stats['max']:,.0f}")
# print(f"   • 75% loans below: ${loan_amount_stats['75%']:,.0f}")

# # 2. High-Risk Segment Analysis
# print("\n2. HIGH-RISK SEGMENTS:")
# high_risk_grade = df[df['grade'].isin(['F','G'])]['default'].mean()
# high_dti = df[df['dti'] > 40]['default'].mean()
# print(f"   • Grade F/G default rate: {high_risk_grade:.2%}")
# print(f"   • DTI > 40 default rate: {high_dti:.2%}")

# # 3. Digital Lending Specific Metrics
# print("\n3. DIGITAL LENDING METRICS:")
# if 'verification_status' in df.columns:
#     verified_risk = df.groupby('verification_status')['default'].mean()
#     print("   • Income verification impact:")
#     for status, rate in verified_risk.items():
#         print(f"     {status}: {rate:.2%}")

# # ACTIONABLE VISUALIZATIONS - Day 3
# print("\nPlotting distributions...")

# # 1. Risk by Loan Grade (Most Important)
# plt.figure(figsize=(10, 6))
# risk_by_grade = df.groupby('grade')['default'].mean()
# risk_by_grade.plot(kind='bar', color=['green','lightgreen','yellow','orange','red','darkred','maroon'])
# plt.title('Default Risk by Loan Grade - Digital Lending Portfolio')
# plt.ylabel('Default Rate')
# plt.xlabel('Loan Grade (A=Lowest Risk, G=Highest Risk)')
# plt.grid(axis='y', alpha=0.3)
# plt.tight_layout()
# plt.savefig('risk_by_grade.png', dpi=300, bbox_inches='tight')
# plt.show()

# # 2. Loan Amount Distribution with Risk Overlay
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# # Loan amount distribution
# ax1.hist(df['loan_amnt'], bins=50, alpha=0.7, color='blue', edgecolor='black')
# ax1.set_title('Loan Amount Distribution')
# ax1.set_xlabel('Loan Amount ($)')
# ax1.set_ylabel('Number of Loans')
# ax1.grid(alpha=0.3)

# # Default rate by loan amount bins
# df['loan_amnt_bin'] = pd.cut(df['loan_amnt'], bins=10)
# default_by_amt = df.groupby('loan_amnt_bin')['default'].mean()
# default_by_amt.plot(kind='bar', ax=ax2, color='red', alpha=0.7)
# ax2.set_title('Default Rate by Loan Amount')
# ax2.set_xlabel('Loan Amount Bins')
# ax2.set_ylabel('Default Rate')
# ax2.tick_params(axis='x', rotation=45)
# ax2.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig('loan_amount_analysis.png', dpi=300, bbox_inches='tight')
# plt.show()

# # 3. Income vs Default Risk
# plt.figure(figsize=(10, 6))
# df['income_k'] = df['annual_inc'] / 1000
# income_risk = df.groupby(pd.cut(df['income_k'], bins=[0, 50, 100, 200, np.inf]))['default'].mean()
# income_risk.plot(kind='bar', color=['red','orange','yellow','green'])
# plt.title('Default Risk by Income Levels')
# plt.xlabel('Annual Income (Thousands $)')
# plt.ylabel('Default Rate')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('income_risk_analysis.png', dpi=300, bbox_inches='tight')
# plt.show()


# DAY 4: BIVARIATE ANALYSIS & EXECUTIVE SUMMARY

# print("\nDAY 4: BIVARIATE ANALYSIS & EXECUTIVE SUMMARY")

# # DIGITAL LENDING RISK SCORE FACTORS
# print("\nRISK SCORING FACTORS ")

# # Calculate risk multipliers
# risk_factors = {}
# for factor in ['grade', 'term', 'home_ownership', 'purpose']:
#     if factor in df.columns:
#         risk_rates = df.groupby(factor)['default'].mean()
#         max_risk = risk_rates.max()
#         min_risk = risk_rates.min()
#         risk_multiplier = max_risk / min_risk if min_risk > 0 else float('inf')
#         risk_factors[factor] = risk_multiplier
#         print(f"   • {factor}: {risk_multiplier:.1f}x risk variation")

# # EXECUTIVE RISK DASHBOARD

# print("DIGITAL LENDING RISK DASHBOARD")


# print(f" PORTFOLIO RISK METRICS")
# print(f"    Overall default rate: {df['default'].mean():.2%}")
# print(f"    High-risk loans (F/G): {len(df[df['grade'].isin(['F','G'])]) / len(df):.2%}")
# print(f"    Concentration risk: {len(df[df['loan_amnt'] == 10000]) / len(df):.2%} at $10,000")

# print(f"\n FINANCIAL EXPOSURE")
# total_exposure = df['loan_amnt'].sum()
# high_risk_exposure = df[df['grade'].isin(['F','G'])]['loan_amnt'].sum()
# print(f" Total portfolio: ${total_exposure:,.0f}")
# print(f" High-risk exposure: ${high_risk_exposure:,.0f}")

# print(f"\n KEY DRIVERS")
# top_corr = df[['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'default']].corr()['default'].sort_values(key=abs, ascending=False)
# print(f"  Strongest predictor: {top_corr.index[1]} ({top_corr.iloc[1]:.3f})")

# print(f"\n RECOMMENDATIONS")
# if high_risk_exposure / total_exposure > 0.1:
#     print(f"   • REDUCE high-grade F/G exposure (currently {high_risk_exposure/total_exposure:.2%})")
# if df['dti'].mean() > 20:
#     print(f"   • REVIEW DTI thresholds (current avg: {df['dti'].mean():.1f})")


#  done by day 4


# print("DAY 5: FEATURE ENGINEERING & RISK SEGMENTATION")

# # Ensure 'term' is numeric
# # Convert term to numeric
# # Convert 'term' to numeric
# df['term'] = df['term'].str.replace(' months','').astype(int)

#     # Derived features
# df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
# df['installment_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)
# df['interest_burden'] = df['installment'] * df['term'] / (df['annual_inc'] + 1)

#     # Income band
# income_bins = [0, 50000, 100000, 200000, float('inf')]
# income_labels = ['Low','Medium','High','Very High']
# df['income_band'] = pd.cut(df['annual_inc'], bins=income_bins, labels=income_labels)

#     # DTI band
# dti_bins = [0, 20, 40, float('inf')]
# dti_labels = ['Safe','Medium Risk','High Risk']
# df['dti_band'] = pd.cut(df['dti'], bins=dti_bins, labels=dti_labels)

#     # Risk segment
# def risk_segment(row):
#         if row['grade'] in ['A','B','C'] and row['dti'] < 20 and row['annual_inc'] > 100000:
#             return 'Low Risk'
#         elif row['grade'] in ['D','E'] and 20 <= row['dti'] <= 40 and 50000 <= row['annual_inc'] <= 100000:
#             return 'Medium Risk'
#         else:
#             return 'High Risk'
# df['risk_segment'] = df.apply(risk_segment, axis=1)

# DAY 6: HYPOTHESIS TESTING & CORRELATION
# ======================
# print("\nDAY 6: HYPOTHESIS TESTING & CORRELATION")

# Step 1: Correlation of numeric features with default
# numeric_cols = [
#     'loan_amnt','int_rate','annual_inc','dti',
#     'loan_to_income','installment_to_income','interest_burden'
# ]
# corr = df[numeric_cols + ['default']].corr()['default'].sort_values(ascending=False)
# print("\nCorrelation of numeric features with default:")
# print(corr)

# Heatmap visualization
# plt.figure(figsize=(8,6))
# sns.heatmap(df[numeric_cols + ['default']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap with Default')
# plt.show()

# Step 2: Hypothesis Testing (ONE main test)


# Make sure int_rate is numeric
# df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')  # safe conversion

# # Separate groups
# defaulted = df[df['default'] == 1]['int_rate'].dropna()
# paid = df[df['default'] == 0]['int_rate'].dropna()

# # Check counts
# print(f"Defaulted count: {len(defaulted)}, Paid count: {len(paid)}")

# # Run t-test
# t_stat, p_val = stats.ttest_ind(defaulted, paid, equal_var=False)

# # Print results
# print("\nHypothesis Test: Interest Rate vs Default")
# print("H0: No difference in interest rates between defaulted and fully paid loans")
# print("H1: Defaulted loans have different interest rates")
# print(f"T-statistic = {t_stat:.2f}, P-value = {p_val:.4f}")


#  day 7 portfolio risk and recommendations
# Step 1: Portfolio Risk Segmentation
# ----------------------
# Create risk_segment first
def risk_segment(row):
    if row['grade'] in ['A','B','C'] and row['dti'] < 20 and row['annual_inc'] > 100000:
        return 'Low Risk'
    elif row['grade'] in ['D','E'] and 20 <= row['dti'] <= 40 and 50000 <= row['annual_inc'] <= 100000:
        return 'Medium Risk'
    else:
        return 'High Risk'

df['risk_segment'] = df.apply(risk_segment, axis=1)

# Then create portfolio summary
portfolio_summary = df.groupby('risk_segment').agg(
    loans_count=('loan_amnt', 'count'),
    total_exposure=('loan_amnt', 'sum'),
    default_rate=('default', 'mean')
).sort_values(by='default_rate', ascending=False)

print("=== Portfolio Risk Summary ===")
print(portfolio_summary)



# Step 2: Exposure by Loan Grade

grade_summary = df.groupby('grade').agg(
    loans_count=('loan_amnt','count'),
    total_exposure=('loan_amnt','sum'),
    default_rate=('default','mean')
).sort_values(by='default_rate', ascending=False)

print("\nExposure by Loan Grade ")
print(grade_summary)

# Step 3: Bar chart visualization (optional)

portfolio_summary['total_exposure'].plot(kind='bar', color=['green','orange','red'], figsize=(8,6))
plt.title('Portfolio Exposure by Risk Segment')
plt.ylabel('Total Exposure ($)')
plt.xlabel('Risk Segment')
plt.xticks(rotation=0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Step 4: Business Recommendations

print("\n BUSINESS RECOMMENDATIONS ")
if portfolio_summary.loc['High Risk','total_exposure'] / portfolio_summary['total_exposure'].sum() > 0.1:
    print
    ("• REDUCE high-grade F/G and High Risk exposure (currently {:.2%})".format(
        portfolio_summary.loc['High Risk','total_exposure'] / portfolio_summary['total_exposure'].sum()))
if df['dti'].mean() > 20:
    print("• REVIEW DTI thresholds (current average DTI: {:.1f})".format(df['dti'].mean()))
if 'verification_status' in df.columns:
    verified_risk = df.groupby('verification_status')['default'].mean()
    print("• Consider stricter verification for high-risk income segments:")
    print(verified_risk)
