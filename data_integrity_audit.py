"""
Data Integrity Audit Script
Checks if Retail Price × Quantity = Sub Total for each row in the pricing dataset
"""

import pandas as pd
import numpy as np

def clean_currency(value):
    """Remove currency symbols and commas from string values"""
    if isinstance(value, str):
        return float(value.replace('$', '').replace(',', ''))
    return float(value)

def audit_data_integrity(file_path):
    """
    Audit the pricing dataset for data integrity issues
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with audit results
    """
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Clean currency columns
    df['Retail Price Clean'] = df['Retail Price'].apply(clean_currency)
    df['Sub Total Clean'] = df['Sub Total'].apply(clean_currency)
    
    # Calculate expected sub total
    df['Expected Sub Total'] = df['Retail Price Clean'] * df['Order Quantity']
    
    # Calculate difference
    df['Difference'] = df['Sub Total Clean'] - df['Expected Sub Total']
    
    # Check if passes audit (allowing for small floating point errors)
    df['Passes Audit'] = np.abs(df['Difference']) < 0.01
    
    # Create audit report
    print("=" * 80)
    print("DATA INTEGRITY AUDIT REPORT")
    print("=" * 80)
    print(f"\nTotal Rows: {len(df)}")
    print(f"Rows Passing Audit: {df['Passes Audit'].sum()}")
    print(f"Rows Failing Audit: {(~df['Passes Audit']).sum()}")
    print(f"Pass Rate: {df['Passes Audit'].sum() / len(df) * 100:.2f}%")
    
    # Show examples of failures
    failures = df[~df['Passes Audit']]
    if len(failures) > 0:
        print("\n" + "=" * 80)
        print("SAMPLE FAILURES (First 10)")
        print("=" * 80)
        
        for idx, row in failures.head(10).iterrows():
            print(f"\nRow {idx + 2} (Order {row['Order No']}):")
            print(f"  Product: {row['Product Name']}")
            print(f"  Retail Price: ${row['Retail Price Clean']:.2f}")
            print(f"  Quantity: {row['Order Quantity']}")
            print(f"  Expected Sub Total: ${row['Expected Sub Total']:.2f}")
            print(f"  Actual Sub Total: ${row['Sub Total Clean']:.2f}")
            print(f"  Difference: ${row['Difference']:.2f}")
    
    # Save detailed audit results
    audit_columns = ['Order No', 'Product Name', 'Retail Price', 'Order Quantity', 
                     'Sub Total', 'Expected Sub Total', 'Difference', 'Passes Audit']
    audit_df = df[audit_columns].copy()
    audit_df['Expected Sub Total'] = audit_df['Expected Sub Total'].apply(lambda x: f"${x:.2f}")
    audit_df['Difference'] = audit_df['Difference'].apply(lambda x: f"${x:.2f}")
    
    output_path = 'data/processed/data_integrity_audit.csv'
    audit_df.to_csv(output_path, index=False)
    print(f"\n\nDetailed audit results saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    file_path = "data/raw/Pricing_dataset.csv"
    audit_df = audit_data_integrity(file_path)
