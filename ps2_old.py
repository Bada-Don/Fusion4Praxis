import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load your dataset
df = pd.read_csv('pricing_dataset.csv')

# Check data types
print("Data Types:")
print(df.dtypes)
print(f"\nShape: {df.shape}")

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

# Remove ID column from analysis (not meaningful for correlation)
if 'product_id' in numerical_cols:
    numerical_cols.remove('product_id')

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# Calculate correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Display correlation matrix
print("\n=== CORRELATION MATRIX ===")
print(correlation_matrix.round(3))

# Visualize with heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5)
plt.title('Correlation Heatmap - Numerical Variables', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

def find_strong_correlations(corr_matrix, threshold=0.5):
    """Find pairs with correlation above threshold"""
    strong_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) >= threshold:
                strong_corr.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'Correlation': round(corr_value, 4),
                    'Strength': 'Strong' if abs(corr_value) >= 0.7 else 'Moderate'
                })
    
    return pd.DataFrame(strong_corr).sort_values('Correlation', 
                                                   key=abs, 
                                                   ascending=False)

# Find correlations >= 0.5
strong_correlations = find_strong_correlations(correlation_matrix, threshold=0.5)
print("\n=== STRONG CORRELATIONS (|r| >= 0.5) ===")
print(strong_correlations.to_string(index=False))

from scipy.stats import chi2_contingency

def cramers_v(x, y):
    """Calculate Cramér's V for categorical variables"""
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = len(x)
    min_dim = min(contingency.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

# Categorical correlation matrix
cat_cols = ['category', 'brand', 'city', 'seller', 'is_organic', 
            'packaging_type', 'offer_type', 'delivery_status']

# Filter existing columns
cat_cols = [col for col in cat_cols if col in df.columns]

cat_corr_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

for col1 in cat_cols:
    for col2 in cat_cols:
        cat_corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

print("\n=== CATEGORICAL CORRELATIONS (Cramér's V) ===")
print(cat_corr_matrix.round(3))

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(cat_corr_matrix.astype(float), annot=True, cmap='YlOrRd', 
            fmt='.2f', square=True)
plt.title("Categorical Variables - Cramér's V")
plt.tight_layout()
plt.show()

# For binary categorical vs numerical
from scipy.stats import pointbiserialr

# Convert is_organic to binary
df['is_organic_binary'] = df['is_organic'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})

print("\n=== POINT-BISERIAL CORRELATIONS (is_organic vs numerical) ===")
for col in numerical_cols:
    if col != 'is_organic_binary':
        corr, pval = pointbiserialr(df['is_organic_binary'], df[col])
        if abs(corr) > 0.1:  # Only show meaningful correlations
            print(f"is_organic vs {col}: r={corr:.3f}, p-value={pval:.4f}")

# Generate summary report
print("\n" + "="*60)
print("CORRELATION ANALYSIS SUMMARY")
print("="*60)
print(f"Total numerical variables analyzed: {len(numerical_cols)}")
print(f"Total categorical variables analyzed: {len(cat_cols)}")
print(f"Strong correlations found (|r| >= 0.7): {len(strong_correlations[strong_correlations['Strength']=='Strong'])}")
print(f"Moderate correlations found (0.5 <= |r| < 0.7): {len(strong_correlations[strong_correlations['Strength']=='Moderate'])}")
