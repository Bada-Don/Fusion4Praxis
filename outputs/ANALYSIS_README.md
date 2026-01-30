# Correlation Analysis Results

## Overview
This directory contains the outputs from `ps2_correlation_analysis.py` - a comprehensive correlation analysis of the Amazon pricing dataset.

## ⚠️ Important Warning
**This dataset is SYNTHETIC with known data quality issues. Results should NOT be used for production ML models.**

## Generated Files

### Visualizations
1. **correlation_heatmap.png** - Correlation matrix heatmap showing relationships between numerical variables
   - Discounted price, actual price, discount percentage, ratings, etc.
   - Color-coded from -1 (negative correlation) to +1 (positive correlation)

2. **categorical_correlation.png** - Cramér's V correlation matrix for categorical variables
   - Shows associations between category, has_rating, and has_reviews

3. **category_analysis.png** - Four-panel visualization showing:
   - Discounted price distribution by category (box plot)
   - Discount percentage by category (box plot)
   - Average rating by category (bar chart)
   - Product count by category (bar chart)

### Data Reports
4. **data_quality_report.csv** - Detailed data quality metrics including:
   - Total products analyzed
   - Products with discount > 100%
   - Products with negative prices
   - Discount calculation errors
   - Missing data counts
   - Duplicate product IDs

## Key Findings

### Strong Correlations Found (|r| >= 0.5)
- **price_difference ↔ discount_amount**: 1.00 (Very Strong) - These are essentially the same metric
- **discounted_price ↔ actual_price**: 0.96 (Very Strong) - Expected relationship
- **actual_price ↔ price_difference**: 0.91 (Very Strong) - Higher priced items have larger absolute discounts
- **actual_price ↔ discount_amount**: 0.91 (Very Strong) - Same as above

### Data Quality Issues Identified
- **114 duplicate product IDs** (7.8% of dataset)
- **1 product missing rating** (0.1%)
- **211 unique categories** - Very fragmented, may need consolidation

## Usage Notes
- All analysis performed on 1,465 products
- Currency values cleaned from ₹ format
- Percentage values cleaned from % format
- Missing values handled appropriately in calculations

## Script Information
- **Script**: `ps2_correlation_analysis.py`
- **Dataset**: `data/raw/Pricing_dataset.csv`
- **Date Generated**: Check file timestamps
- **Python Libraries**: pandas, numpy, matplotlib, seaborn, scipy
