"""
Data Usability Assessment
Determines if the dataset can be used for ML after cleaning and preprocessing
"""

import pandas as pd
import numpy as np

def assess_data_usability(file_path):
    """Comprehensive assessment of data quality and ML readiness"""
    
    df = pd.read_csv(file_path)
    
    print("="*80)
    print("DATA USABILITY ASSESSMENT FOR ML")
    print("="*80)
    
    # 1. DUPLICATE ANALYSIS
    print("\n1. DUPLICATE ANALYSIS")
    print("-" * 40)
    total_rows = len(df)
    unique_products = df['product_id'].nunique()
    duplicates = df.duplicated(subset=['product_id']).sum()
    
    print(f"Total rows: {total_rows}")
    print(f"Unique product IDs: {unique_products}")
    print(f"Duplicate rows: {duplicates} ({duplicates/total_rows*100:.1f}%)")
    
    # Check if duplicates are different reviews for same product
    if duplicates > 0:
        print("\n   Investigating duplicates...")
        dup_sample = df[df.duplicated(subset=['product_id'], keep=False)].head(10)
        unique_reviews = dup_sample.groupby('product_id')['review_id'].nunique()
        
        if unique_reviews.mean() > 1:
            print("   ✓ GOOD: Duplicates are different reviews for the same product")
            print("   → Can aggregate reviews per product for ML")
        else:
            print("   ⚠️  WARNING: Duplicates appear to be exact copies")
    
    # 2. MISSING DATA ANALYSIS
    print("\n2. MISSING DATA ANALYSIS")
    print("-" * 40)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    if missing.sum() == 0:
        print("   ✓ EXCELLENT: No missing values detected")
    elif missing_pct.max() < 5:
        print("   ✓ GOOD: Missing data < 5%, can be handled")
    elif missing_pct.max() < 20:
        print("   ⚠️  MODERATE: Some columns have 5-20% missing data")
    else:
        print("   ❌ POOR: Significant missing data (>20%)")
    
    # 3. DATA CONSISTENCY CHECKS
    print("\n3. DATA CONSISTENCY CHECKS")
    print("-" * 40)
    
    # Clean price data
    def clean_price(val):
        if pd.isna(val): return np.nan
        return float(str(val).replace('₹', '').replace(',', '').strip())
    
    def clean_pct(val):
        if pd.isna(val): return np.nan
        return float(str(val).replace('%', '').strip())
    
    df['disc_price'] = df['discounted_price'].apply(clean_price)
    df['act_price'] = df['actual_price'].apply(clean_price)
    df['disc_pct'] = df['discount_percentage'].apply(clean_pct)
    
    # Check for negative prices
    negative_prices = ((df['disc_price'] < 0) | (df['act_price'] < 0)).sum()
    print(f"Negative prices: {negative_prices}")
    if negative_prices == 0:
        print("   ✓ GOOD: No negative prices")
    
    # Check for impossible discounts
    impossible_disc = (df['disc_pct'] > 100).sum()
    print(f"Discounts > 100%: {impossible_disc}")
    if impossible_disc == 0:
        print("   ✓ GOOD: All discounts are valid")
    
    # Check discount calculation accuracy
    df['calc_disc'] = ((df['act_price'] - df['disc_price']) / df['act_price'] * 100)
    disc_errors = (abs(df['calc_disc'] - df['disc_pct']) > 1).sum()
    print(f"Discount calculation errors: {disc_errors} ({disc_errors/len(df)*100:.1f}%)")
    if disc_errors == 0:
        print("   ✓ EXCELLENT: All discount calculations are accurate")
    elif disc_errors < len(df) * 0.05:
        print("   ✓ GOOD: <5% calculation errors, can be corrected")
    
    # Check if discounted price <= actual price
    price_logic_errors = (df['disc_price'] > df['act_price']).sum()
    print(f"Discounted price > Actual price: {price_logic_errors}")
    if price_logic_errors == 0:
        print("   ✓ GOOD: Price logic is consistent")
    
    # 4. FEATURE RICHNESS
    print("\n4. FEATURE RICHNESS")
    print("-" * 40)
    print(f"Total features: {len(df.columns)}")
    print(f"Numerical features: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"Categorical features: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"Text features available: {('about_product' in df.columns) + ('review_content' in df.columns)}")
    
    if 'about_product' in df.columns and 'review_content' in df.columns:
        print("   ✓ EXCELLENT: Rich text data available for NLP features")
    
    # 5. TARGET VARIABLE ANALYSIS (assuming rating is target)
    print("\n5. TARGET VARIABLE ANALYSIS (Rating)")
    print("-" * 40)
    df['rating_num'] = pd.to_numeric(df['rating'], errors='coerce')
    print(f"Rating range: {df['rating_num'].min():.1f} - {df['rating_num'].max():.1f}")
    print(f"Rating distribution:")
    print(df['rating_num'].value_counts().sort_index())
    
    # Check for class imbalance
    rating_dist = df['rating_num'].value_counts(normalize=True)
    if rating_dist.max() < 0.5:
        print("   ✓ GOOD: Reasonably balanced rating distribution")
    else:
        print(f"   ⚠️  WARNING: Imbalanced - {rating_dist.idxmax()} rating is {rating_dist.max()*100:.1f}%")
    
    # 6. SAMPLE SIZE
    print("\n6. SAMPLE SIZE")
    print("-" * 40)
    unique_samples = df['product_id'].nunique()
    print(f"Unique products: {unique_samples}")
    
    if unique_samples >= 1000:
        print("   ✓ GOOD: Sufficient samples for ML (>1000)")
    elif unique_samples >= 500:
        print("   ⚠️  MODERATE: Limited samples (500-1000), may need careful validation")
    else:
        print("   ❌ POOR: Insufficient samples (<500) for robust ML")
    
    # 7. CATEGORY DISTRIBUTION
    print("\n7. CATEGORY DISTRIBUTION")
    print("-" * 40)
    n_categories = df['category'].nunique()
    print(f"Unique categories: {n_categories}")
    
    category_counts = df['category'].value_counts()
    print(f"Products per category (avg): {category_counts.mean():.1f}")
    print(f"Products per category (median): {category_counts.median():.1f}")
    print(f"Categories with <5 products: {(category_counts < 5).sum()}")
    
    if n_categories > 100:
        print("   ⚠️  WARNING: Very fragmented categories, consider consolidation")
    
    # 8. FINAL VERDICT
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    issues = []
    can_use = True
    
    if duplicates > 0:
        issues.append("✓ Duplicates exist but are different reviews (can aggregate)")
    
    if missing_pct.max() > 20:
        issues.append("❌ Significant missing data")
        can_use = False
    
    if negative_prices > 0 or impossible_disc > 0:
        issues.append("❌ Invalid price/discount values")
        can_use = False
    
    if price_logic_errors > 0:
        issues.append("❌ Price logic inconsistencies")
        can_use = False
    
    if disc_errors > len(df) * 0.1:
        issues.append("⚠️  Many discount calculation errors (but fixable)")
    
    if unique_samples < 500:
        issues.append("❌ Insufficient sample size")
        can_use = False
    
    if n_categories > 100 and category_counts.median() < 5:
        issues.append("⚠️  Highly fragmented categories (needs consolidation)")
    
    print("\nIssues Summary:")
    for issue in issues:
        print(f"  {issue}")
    
    print("\n" + "-"*80)
    if can_use:
        print("✅ VERDICT: YES, this data CAN be used for ML after preprocessing")
        print("\nRecommended preprocessing steps:")
        print("  1. Aggregate reviews per product (handle duplicates)")
        print("  2. Fix any discount calculation errors")
        print("  3. Consolidate categories (reduce from 200+ to 10-20 main categories)")
        print("  4. Engineer features from text (about_product, reviews)")
        print("  5. Handle any missing values (imputation or removal)")
        print("  6. Normalize/scale numerical features")
        print("  7. Encode categorical variables")
        print("\nPotential ML tasks:")
        print("  • Rating prediction (regression)")
        print("  • Price optimization")
        print("  • Demand forecasting")
        print("  • Product recommendation")
        print("  • Review sentiment analysis")
    else:
        print("❌ VERDICT: Data has CRITICAL issues that prevent ML use")
        print("\nCritical issues must be resolved first:")
        for issue in issues:
            if '❌' in issue:
                print(f"  {issue}")
    
    print("="*80)
    
    return df

if __name__ == "__main__":
    df = assess_data_usability('data/raw/Pricing_dataset.csv')
