import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob

# --- Phase 1: Data Loading and Feature Engineering ---
@st.cache_data
def load_and_prepare_data():
    # Load data
    df = pd.read_csv("data/raw/Pricing_dataset.csv")
    
    # Cleaning Money Columns
    def clean_currency(x):
        if isinstance(x, str):
            x = x.replace('₹', '').replace(',', '').strip()
        return pd.to_numeric(x, errors='coerce')
    
    df['discounted_price'] = df['discounted_price'].apply(clean_currency)
    df['actual_price'] = df['actual_price'].apply(clean_currency)
    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
    
    # Cleaning Rating/Demand Columns
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Drop nulls
    df.dropna(subset=['discounted_price', 'actual_price', 'rating_count', 'rating'], inplace=True)
    
    # Category Hierarchy Processing
    df['cat_split'] = df['category'].str.split('|')
    df['main_category'] = df['cat_split'].apply(lambda x: x[0] if isinstance(x, list) else "Other")
    df['sub_category'] = df['cat_split'].apply(lambda x: x[-1] if isinstance(x, list) else "Other")
    df.drop('cat_split', axis=1, inplace=True)  # Drop to avoid caching issues
    
    # Visibility Metrics (Content Quality)
    df['desc_len'] = df['about_product'].astype(str).apply(len)
    df['name_len'] = df['product_name'].astype(str).apply(len)
    
    # Relative Pricing (Context)
    df['category_avg_price'] = df.groupby('sub_category')['discounted_price'].transform('mean')
    df['price_competitiveness'] = df['discounted_price'] / df['category_avg_price']
    
    # Sentiment Analysis
    def get_sentiment(text):
        if pd.isna(text):
            return 0
        return TextBlob(str(text)).sentiment.polarity
    
    df['review_sentiment'] = df['review_content'].apply(get_sentiment)
    
    # Target Transformation
    df['log_demand'] = np.log1p(df['rating_count'])
    
    return df

# --- Phase 2: Model Training ---
@st.cache_resource
def train_model(_df):
    # Encode Categoricals
    le = LabelEncoder()
    _df['main_category_encoded'] = le.fit_transform(_df['main_category'])
    
    # Select Features
    features = [
        'discounted_price', 
        'actual_price', 
        'discount_percentage', 
        'rating',
        'desc_len',
        'price_competitiveness',
        'review_sentiment',
        'main_category_encoded'
    ]
    
    X = _df[features]
    y = _df['log_demand']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost Regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=6
    )
    
    model.fit(X_train, y_train)
    
    return model, le

# Load data and train model
df = load_and_prepare_data()
model, label_encoder = train_model(df)

# Add encoded category to df (needed for product selection)
df['main_category_encoded'] = label_encoder.transform(df['main_category'])

st.title("🛒 Dynamic Pricing & Demand Simulator")
st.markdown("### Optimize Price & Visibility to Maximize Revenue")

# Sidebar for User Inputs
st.sidebar.header("Scenario Settings")

# Pick a product to simulate
st.sidebar.subheader("Product Selection")
selection_method = st.sidebar.radio("Choose method:", ["Quick Select", "Custom Index"])

if selection_method == "Quick Select":
    product_idx = st.sidebar.selectbox("Select Product Index", options=[10, 50, 100, 200, 500, 1000])
else:
    max_index = len(df) - 1
    product_idx = st.sidebar.number_input(
        f"Enter Product Index (0 to {max_index})", 
        min_value=0, 
        max_value=max_index, 
        value=10,
        step=1
    )

selected_product = df.iloc[product_idx]
st.sidebar.write(f"**Product:** {selected_product['product_name'][:50]}...")
st.sidebar.write(f"**Category:** {selected_product['main_category']}")
st.sidebar.write(f"**Current Price:** ₹{selected_product['discounted_price']:.0f}")
st.sidebar.write(f"**Rating:** {selected_product['rating']:.1f} ⭐")

st.sidebar.markdown("---")

# Sliders for "What-If" Analysis
price_delta = st.sidebar.slider("Change Price (%)", -50, 50, 0)
visibility_delta = st.sidebar.slider("Improve Description/SEO (%)", 0, 100, 0)

# Product Details Section
with st.expander("📦 Product Details"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Product Name:** {selected_product['product_name']}")
        st.write(f"**Category:** {selected_product['category']}")
        st.write(f"**Current Sales (proxy):** {int(np.expm1(selected_product['log_demand']))} units")
    with col_b:
        st.write(f"**Actual Price:** ₹{selected_product['actual_price']:.0f}")
        st.write(f"**Discounted Price:** ₹{selected_product['discounted_price']:.0f}")
        st.write(f"**Rating:** {selected_product['rating']:.1f} ({int(selected_product['rating_count'])} reviews)")
        st.write(f"**Sentiment Score:** {selected_product['review_sentiment']:.2f}")

# --- SIMULATION LOGIC ---
current_price = selected_product['discounted_price']
current_desc_len = selected_product['desc_len']
avg_cat_price = selected_product['category_avg_price']

# Apply changes
new_price = current_price * (1 + price_delta/100)
new_desc_len = current_desc_len * (1 + visibility_delta/100)
new_competitiveness = new_price / avg_cat_price

# Prepare input for model
input_data = pd.DataFrame([{
    'discounted_price': new_price,
    'actual_price': selected_product['actual_price'],
    'discount_percentage': ((selected_product['actual_price'] - new_price) / selected_product['actual_price']) * 100,
    'rating': selected_product['rating'],
    'desc_len': new_desc_len,
    'price_competitiveness': new_competitiveness,
    'review_sentiment': selected_product['review_sentiment'],
    'main_category_encoded': selected_product['main_category_encoded']
}])

# Predict
predicted_log_demand = model.predict(input_data)[0]
predicted_demand = np.expm1(predicted_log_demand)

# Calculate baseline (current state)
baseline_demand = np.expm1(selected_product['log_demand'])
baseline_revenue = current_price * baseline_demand
new_revenue = new_price * predicted_demand

# Show Results
st.subheader("📊 Scenario Results")
col1, col2, col3 = st.columns(3)
col1.metric("New Price", f"₹{new_price:.0f}", f"{price_delta:+.0f}%")
col2.metric("Predicted Sales Volume", f"{int(predicted_demand)}", f"{int(predicted_demand - baseline_demand):+.0f} units")
col3.metric("Projected Revenue", f"₹{new_revenue:,.0f}", f"{((new_revenue - baseline_revenue) / baseline_revenue * 100):+.1f}%")

# Visuals
st.subheader("📈 Price vs. Demand Sensitivity Curve")
st.markdown("*This curve shows how demand changes with price adjustments*")

# Generate a quick curve
prices = np.linspace(current_price * 0.5, current_price * 1.5, 20)
demands = []
revenues = []
for p in prices:
    temp_input = input_data.copy()
    temp_input['discounted_price'] = p
    temp_input['price_competitiveness'] = p / avg_cat_price
    temp_input['discount_percentage'] = ((selected_product['actual_price'] - p) / selected_product['actual_price']) * 100
    demand = np.expm1(model.predict(temp_input)[0])
    demands.append(demand)
    revenues.append(p * demand)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Demand Curve
ax1.plot(prices, demands, color='green', linewidth=2)
ax1.axvline(new_price, color='red', linestyle='--', label='Selected Price', linewidth=2)
ax1.axvline(current_price, color='blue', linestyle='--', label='Current Price', linewidth=2)
ax1.set_xlabel("Price (₹)", fontsize=11)
ax1.set_ylabel("Predicted Sales Volume", fontsize=11)
ax1.set_title("Price vs. Demand", fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Revenue Curve
ax2.plot(prices, revenues, color='purple', linewidth=2)
ax2.axvline(new_price, color='red', linestyle='--', label='Selected Price', linewidth=2)
ax2.axvline(current_price, color='blue', linestyle='--', label='Current Price', linewidth=2)
ax2.set_xlabel("Price (₹)", fontsize=11)
ax2.set_ylabel("Projected Revenue (₹)", fontsize=11)
ax2.set_title("Price vs. Revenue", fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Key Insights
st.subheader("💡 Key Insights")
optimal_price_idx = np.argmax(revenues)
optimal_price = prices[optimal_price_idx]
optimal_revenue = revenues[optimal_price_idx]
optimal_demand = demands[optimal_price_idx]

col_i1, col_i2, col_i3, col_i4 = st.columns(4)
with col_i1:
    st.metric("Optimal Price Point", f"₹{optimal_price:.0f}", help="Price that maximizes revenue")
with col_i2:
    st.metric("Baseline Sales", f"{int(baseline_demand)}", help="Current sales volume at current price")
with col_i3:
    st.metric("Sales at Optimal Price", f"{int(optimal_demand)}", f"{int(optimal_demand - baseline_demand):+.0f} units", help="Predicted sales if price is changed to optimal")
with col_i4:
    st.metric("Max Potential Revenue", f"₹{optimal_revenue:,.0f}", help=f"Revenue at optimal price: ₹{optimal_price:.0f} × {int(optimal_demand)} units")

st.markdown("---")
st.caption("Built with XGBoost | Data-driven pricing optimization for Amazon products")