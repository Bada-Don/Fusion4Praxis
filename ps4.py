# Correlation Analysis for Student Performance Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('student_dataset.csv')

# Basic info
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())


# Numerical columns
numerical_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                  'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 
                  'absences', 'G1', 'G2', 'G3']

# Categorical columns
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 
                    'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid',
                    'activities', 'nursery', 'higher', 'internet', 'romantic', 
                    'part_time_job']

# Binary columns (yes/no)
binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
               'higher', 'internet', 'romantic', 'part_time_job']

print(f"Numerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")


# Calculate correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Create heatmap
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Upper triangle mask
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            annot_kws={'size': 8})
plt.title('Correlation Heatmap - Student Performance Dataset', fontsize=14)
plt.tight_layout()
plt.savefig('student_correlation_heatmap.png', dpi=300)
plt.show()


def find_strong_correlations(corr_matrix, threshold=0.3):
    """Find pairs with correlation above threshold"""
    strong_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) >= threshold:
                # Determine strength
                if abs(corr_value) >= 0.7:
                    strength = 'Very Strong'
                elif abs(corr_value) >= 0.5:
                    strength = 'Strong'
                else:
                    strength = 'Moderate'
                    
                strong_corr.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'Correlation': round(corr_value, 4),
                    'Strength': strength,
                    'Direction': 'Positive' if corr_value > 0 else 'Negative'
                })
    
    return pd.DataFrame(strong_corr).sort_values('Correlation', 
                                                   key=abs, 
                                                   ascending=False)

# Find correlations >= 0.3
strong_correlations = find_strong_correlations(correlation_matrix, threshold=0.3)
print("\n" + "="*70)
print("STRONG CORRELATIONS (|r| >= 0.3)")
print("="*70)
print(strong_correlations.to_string(index=False))


# Focus on what correlates with final grade (G3)
print("\n" + "="*70)
print("CORRELATIONS WITH FINAL GRADE (G3)")
print("="*70)

g3_correlations = correlation_matrix['G3'].drop('G3').sort_values(key=abs, ascending=False)

# Create bar plot
plt.figure(figsize=(12, 8))
colors = ['green' if x > 0 else 'red' for x in g3_correlations.values]
bars = plt.barh(g3_correlations.index, g3_correlations.values, color=colors, alpha=0.7)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Variables')
plt.title('Correlation of Variables with Final Grade (G3)')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.axvline(x=0.3, color='gray', linestyle='--', linewidth=0.5, label='Threshold ±0.3')
plt.axvline(x=-0.3, color='gray', linestyle='--', linewidth=0.5)

# Add value labels
for bar, val in zip(bars, g3_correlations.values):
    plt.text(val + 0.02 if val > 0 else val - 0.08, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('g3_correlations.png', dpi=300)
plt.show()

# Print as table
print("\nVariable".ljust(15) + "Correlation".ljust(15) + "Impact on Grade")
print("-"*50)
for var, corr in g3_correlations.items():
    impact = "Positive ↑" if corr > 0.1 else ("Negative ↓" if corr < -0.1 else "Minimal")
    print(f"{var.ljust(15)}{str(round(corr, 3)).ljust(15)}{impact}")


# Convert binary columns to 0/1
for col in binary_cols:
    if col in df.columns:
        df[f'{col}_binary'] = df[col].map({'yes': 1, 'no': 0})

# Point-biserial correlation for binary vs G3
print("\n" + "="*70)
print("BINARY VARIABLES vs FINAL GRADE (G3)")
print("="*70)

binary_correlations = []
for col in binary_cols:
    if f'{col}_binary' in df.columns:
        corr, pval = pointbiserialr(df[f'{col}_binary'].dropna(), 
                                     df.loc[df[f'{col}_binary'].notna(), 'G3'])
        binary_correlations.append({
            'Variable': col,
            'Correlation': round(corr, 4),
            'P-value': round(pval, 4),
            'Significant': '✓' if pval < 0.05 else '✗'
        })

binary_df = pd.DataFrame(binary_correlations).sort_values('Correlation', 
                                                            key=abs, 
                                                            ascending=False)
print(binary_df.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in binary_df['Correlation']]
plt.barh(binary_df['Variable'], binary_df['Correlation'], color=colors, alpha=0.7)
plt.xlabel('Point-Biserial Correlation')
plt.title('Binary Variables vs Final Grade (G3)')
plt.axvline(x=0, color='black', linestyle='-')
plt.tight_layout()
plt.show()


def cramers_v(x, y):
    """Calculate Cramér's V for categorical variables"""
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = len(x)
    min_dim = min(contingency.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

# Select key categorical columns
key_cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                'Mjob', 'Fjob', 'higher', 'internet', 'romantic']

# Calculate Cramér's V matrix
cat_corr = pd.DataFrame(index=key_cat_cols, columns=key_cat_cols, dtype=float)
for col1 in key_cat_cols:
    for col2 in key_cat_cols:
        cat_corr.loc[col1, col2] = cramers_v(df[col1], df[col2])

plt.figure(figsize=(12, 10))
sns.heatmap(cat_corr.astype(float), annot=True, cmap='YlOrRd', 
            fmt='.2f', square=True, linewidths=0.5)
plt.title("Categorical Variables Correlation (Cramér's V)")
plt.tight_layout()
plt.savefig('categorical_correlation.png', dpi=300)
plt.show()


# Special focus on alcohol patterns
print("\n" + "="*70)
print("ALCOHOL CONSUMPTION CORRELATIONS")
print("="*70)

alcohol_vars = ['Dalc', 'Walc']
for alc in alcohol_vars:
    print(f"\n{alc} correlations:")
    alc_corr = correlation_matrix[alc].drop(alcohol_vars).sort_values(key=abs, ascending=False)
    for var, corr in alc_corr.head(5).items():
        print(f"  {var}: {corr:.3f}")


print("\n" + "="*70)
print("CORRELATION ANALYSIS SUMMARY - STUDENT PERFORMANCE")
print("="*70)

print("\n📊 KEY FINDINGS:")
print("-"*50)

# Top positive correlations
print("\n✅ STRONGEST POSITIVE CORRELATIONS:")
positive = strong_correlations[strong_correlations['Direction'] == 'Positive'].head(5)
for _, row in positive.iterrows():
    print(f"   • {row['Variable 1']} ↔ {row['Variable 2']}: {row['Correlation']}")

# Top negative correlations  
print("\n❌ STRONGEST NEGATIVE CORRELATIONS:")
negative = strong_correlations[strong_correlations['Direction'] == 'Negative'].head(5)
for _, row in negative.iterrows():
    print(f"   • {row['Variable 1']} ↔ {row['Variable 2']}: {row['Correlation']}")

# Factors affecting grades
print("\n🎓 TOP FACTORS AFFECTING FINAL GRADE (G3):")
for var, corr in g3_correlations.head(5).items():
    direction = "improves" if corr > 0 else "decreases"
    print(f"   • Higher {var} → {direction} grades (r={corr:.3f})")

print("\n" + "="*70)