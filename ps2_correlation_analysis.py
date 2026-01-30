"""
Correlation Analysis for Amazon Pricing Dataset
WARNING: This dataset is synthetic with known data quality issues.
Results should be interpreted with caution and not used for production ML models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def clean_currency(value):
    """Remove currency symbols and commas from string values"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        return float(value.replace('₹', '').replace(',', '').strip())
    return float(value)

def clean_percentage(value):
    """Remove percentage symbol and convert to float"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        return float(value.replace('%', '').strip())
    return float(value)

def load_and_clean_data(file_path):
    """Load and clean the dataset"""
    print("="*80)
    print("LOADING AND CLEANING DATA")
    print("="*80)
    
    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    
    # Clean numerical columns
    df['discounted_price_clean'] = df['discounted_price'].apply(clean_currency)
    df['actual_price_clean'] = df['actual_price'].apply(clean_currency)
    df['discount_percentage_clean'] = df['discount_percentage'].apply(clean_percentage)
    df['rating_clean'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count_clean'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce')
    
    # Calculate additional metrics
    df['price_difference'] = df['actual_price_clean'] - df['discounted_price_clean']
    df['discount_amount'] = df['actual_price_clean'] * (df['discount_percentage_clean'] / 100)
    
    # Data quality flags
    df['has_rating'] = df['rating_clean'].notna()
    df['has_reviews'] = df['rating_count_clean'] > 0
    
    print(f"\nCleaned shape: {df.shape}")
    print(f"Missing values in key columns:")
    print(f"  - Discounted Price: {df['discounted_price_clean'].isna().sum()}")
    print(f"  - Actual Price: {df['actual_price_clean'].isna().sum()}")
    print(f"  - Discount %: {df['discount_percentage_clean'].isna().sum()}")
    print(f"  - Rating: {df['rating_clean'].isna().sum()}")
    print(f"  - Rating Count: {df['rating_count_clean'].isna().sum()}")
    
    return df

def analyze_numerical_correlations(df):
    """Analyze correlations between numerical variables"""
    print("\n" + "="*80)
    print("NUMERICAL CORRELATION ANALYSIS")
    print("="*80)
    
    numerical_cols = [
        'discounted_price_clean',
        'actual_price_clean',
        'discount_percentage_clean',
        'rating_clean',
        'rating_count_clean',
        'price_difference',
        'discount_amount'
    ]
    
    # Filter to existing columns with data
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    # Calculate correlation matrix
    correlation_matrix = df[numerical_cols].corr()
    
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(3))
    
    # Visualize with heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                vmin=-1, vmax=1)
    plt.title('Correlation Heatmap - Numerical Variables\n(Synthetic Dataset - Use with Caution)', 
              fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nHeatmap saved to: outputs/correlation_heatmap.png")
    plt.close()
    
    return correlation_matrix

def find_strong_correlations(corr_matrix, threshold=0.5):
    """Find pairs with correlation above threshold"""
    strong_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) >= threshold:
                strength = 'Very Strong' if abs(corr_value) >= 0.9 else \
                          'Strong' if abs(corr_value) >= 0.7 else 'Moderate'
                strong_corr.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'Correlation': round(corr_value, 4),
                    'Strength': strength
                })
    
    return pd.DataFrame(strong_corr).sort_values('Correlation', 
                                                   key=abs, 
                                                   ascending=False)

def cramers_v(x, y):
    """Calculate Cramér's V for categorical variables"""
    try:
        contingency = pd.crosstab(x, y)
        chi2 = chi2_contingency(contingency)[0]
        n = len(x)
        min_dim = min(contingency.shape) - 1
        return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    except:
        return np.nan

def analyze_categorical_correlations(df):
    """Analyze correlations between categorical variables"""
    print("\n" + "="*80)
    print("CATEGORICAL CORRELATION ANALYSIS (Cramér's V)")
    print("="*80)
    
    cat_cols = ['category']
    
    # Add binary flags if they exist
    if 'has_rating' in df.columns:
        cat_cols.append('has_rating')
    if 'has_reviews' in df.columns:
        cat_cols.append('has_reviews')
    
    if len(cat_cols) < 2:
        print("Not enough categorical variables for analysis")
        return None
    
    cat_corr_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    
    for col1 in cat_cols:
        for col2 in cat_cols:
            cat_corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
    
    print("\nCramér's V Matrix:")
    print(cat_corr_matrix.round(3))
    
    # Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(cat_corr_matrix.astype(float), 
                annot=True, 
                cmap='YlOrRd', 
                fmt='.2f', 
                square=True,
                vmin=0, vmax=1)
    plt.title("Categorical Variables - Cramér's V\n(Synthetic Dataset - Use with Caution)", 
              fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('outputs/categorical_correlation.png', dpi=300, bbox_inches='tight')
    print("\nCategorical correlation plot saved to: outputs/categorical_correlation.png")
    plt.close()
    
    return cat_corr_matrix

def analyze_category_price_relationship(df):
    """Analyze relationship between category and pricing"""
    print("\n" + "="*80)
    print("CATEGORY vs PRICE ANALYSIS")
    print("="*80)
    
    category_stats = df.groupby('category').agg({
        'discounted_price_clean': ['mean', 'median', 'std', 'count'],
        'discount_percentage_clean': ['mean', 'median'],
        'rating_clean': 'mean'
    }).round(2)
    
    print("\nPrice Statistics by Category:")
    print(category_stats)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Box plot for discounted prices
    df.boxplot(column='discounted_price_clean', by='category', ax=axes[0, 0])
    axes[0, 0].set_title('Discounted Price Distribution by Category')
    axes[0, 0].set_xlabel('Category')
    axes[0, 0].set_ylabel('Discounted Price (₹)')
    plt.sca(axes[0, 0])
    plt.xticks(rotation=45, ha='right')
    
    # Box plot for discount percentage
    df.boxplot(column='discount_percentage_clean', by='category', ax=axes[0, 1])
    axes[0, 1].set_title('Discount Percentage by Category')
    axes[0, 1].set_xlabel('Category')
    axes[0, 1].set_ylabel('Discount %')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=45, ha='right')
    
    # Bar plot for average rating
    category_rating = df.groupby('category')['rating_clean'].mean().sort_values(ascending=False)
    category_rating.plot(kind='bar', ax=axes[1, 0], color='skyblue')
    axes[1, 0].set_title('Average Rating by Category')
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Average Rating')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Count plot
    category_counts = df['category'].value_counts()
    category_counts.plot(kind='bar', ax=axes[1, 1], color='coral')
    axes[1, 1].set_title('Product Count by Category')
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Category Analysis\n(Synthetic Dataset - Use with Caution)', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('outputs/category_analysis.png', dpi=300, bbox_inches='tight')
    print("\nCategory analysis plot saved to: outputs/category_analysis.png")
    plt.close()
    
    return category_stats

def data_quality_report(df):
    """Generate data quality report highlighting issues"""
    print("\n" + "="*80)
    print("DATA QUALITY ISSUES REPORT")
    print("="*80)
    print("\n⚠️  Note: Dataset requires cleaning and preprocessing before ML use\n")
    
    issues = []
    
    # Check for impossible discounts
    impossible_discounts = df[df['discount_percentage_clean'] > 100]
    if len(impossible_discounts) > 0:
        issues.append(f"❌ {len(impossible_discounts)} products with discount > 100%")
    
    # Check for negative prices
    negative_prices = df[(df['discounted_price_clean'] < 0) | (df['actual_price_clean'] < 0)]
    if len(negative_prices) > 0:
        issues.append(f"❌ {len(negative_prices)} products with negative prices")
    
    # Check discount calculation accuracy
    df['calculated_discount'] = ((df['actual_price_clean'] - df['discounted_price_clean']) / 
                                  df['actual_price_clean'] * 100)
    discount_mismatch = df[abs(df['calculated_discount'] - df['discount_percentage_clean']) > 1]
    if len(discount_mismatch) > 0:
        issues.append(f"❌ {len(discount_mismatch)} products with discount calculation errors")
    
    # Check for missing critical data
    missing_price = df['discounted_price_clean'].isna().sum()
    if missing_price > 0:
        issues.append(f"❌ {missing_price} products missing discounted price")
    
    missing_rating = df['rating_clean'].isna().sum()
    if missing_rating > 0:
        issues.append(f"⚠️  {missing_rating} products missing ratings ({missing_rating/len(df)*100:.1f}%)")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['product_id']).sum()
    if duplicates > 0:
        issues.append(f"❌ {duplicates} duplicate product IDs")
    
    if issues:
        print("Issues Found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ No major data quality issues detected (but dataset is still synthetic)")
    
    # Save quality report
    quality_df = pd.DataFrame({
        'Metric': [
            'Total Products',
            'Products with Discount > 100%',
            'Products with Negative Prices',
            'Discount Calculation Errors',
            'Missing Discounted Price',
            'Missing Ratings',
            'Duplicate Product IDs'
        ],
        'Count': [
            len(df),
            len(impossible_discounts),
            len(negative_prices),
            len(discount_mismatch),
            missing_price,
            missing_rating,
            duplicates
        ],
        'Percentage': [
            100.0,
            len(impossible_discounts)/len(df)*100,
            len(negative_prices)/len(df)*100,
            len(discount_mismatch)/len(df)*100,
            missing_price/len(df)*100,
            missing_rating/len(df)*100,
            duplicates/len(df)*100
        ]
    })
    
    quality_df.to_csv('outputs/data_quality_report.csv', index=False)
    print(f"\nDetailed quality report saved to: outputs/data_quality_report.csv")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("AMAZON PRICING DATASET - CORRELATION ANALYSIS")
    print("="*80)
    print("Note: Dataset requires preprocessing before ML use")
    print("="*80 + "\n")
    
    # Load and clean data
    df = load_and_clean_data('data/raw/Pricing_dataset.csv')
    
    # Numerical correlations
    corr_matrix = analyze_numerical_correlations(df)
    
    # Find strong correlations
    strong_corr = find_strong_correlations(corr_matrix, threshold=0.5)
    print("\n" + "="*80)
    print("STRONG CORRELATIONS (|r| >= 0.5)")
    print("="*80)
    if len(strong_corr) > 0:
        print(strong_corr.to_string(index=False))
    else:
        print("No strong correlations found")
    
    # Categorical correlations
    analyze_categorical_correlations(df)
    
    # Category-price relationship
    analyze_category_price_relationship(df)
    
    # Data quality report
    data_quality_report(df)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"✓ Analyzed {len(df)} products across {df['category'].nunique()} categories")
    print(f"✓ Generated correlation matrices and visualizations")
    print(f"✓ Identified data quality issues")
    print(f"✓ All outputs saved to 'outputs/' directory")
    print("\n✅ Dataset CAN be used for ML after proper preprocessing:")
    print("   • Aggregate duplicate reviews per product")
    print("   • Consolidate fragmented categories")
    print("   • Engineer features from text data")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    import os
    os.makedirs('outputs', exist_ok=True)
    
    main()
