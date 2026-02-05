import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob

# --- UI Setup ---
st.set_page_config(layout="wide", page_title="Pricing Simulator")
plt.style.use('dark_background')

# --- Custom CSS for Dark Theme & Iframe Support ---
st.markdown("""
    <style>
        /* Remove default Streamlit header, footer, and menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Background - Match Next.js Dark Theme (#020617) */
        .stApp {
            background-color: #020617;
        }
        
        /* Metric Cards Styling */
        div[data-testid="stMetric"] {
            background-color: #1e293b; /* Slate-800 */
            border: 1px solid #334155; /* Slate-700 */
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important; /* Slate-400 */
            font-size: 0.85rem !important;
        }
        
        div[data-testid="stMetricValue"] {
            color: #f8fafc !important; /* Slate-50 */
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a; /* Slate-900 */
            border-right: 1px solid #1e293b;
        }
        
        h1, h2, h3 {
            color: #f8fafc !important;
        }
        
        p, span, label {
            color: #cbd5e1 !important; /* Slate-300 */
        }
    </style>
""", unsafe_allow_html=True)
# --- Phase 1: Data Loading and Feature Engineering ---
@st.cache_data
def load_and_prepare_data():
    # Load data
    df = pd.read_csv("data/Pricing_dataset.csv")
    
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

# --- Phase 2: Model Training (UPDATED WITH CONSTRAINTS) ---
@st.cache_resource
def train_model(_df):
    # Encode Categoricals
    le = LabelEncoder()
    _df['main_category_encoded'] = le.fit_transform(_df['main_category'])
    
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
    
    # --- HERE IS THE MAGIC FIX ---
    # 1 = Increasing constraint (As feature goes up, prediction goes up)
    # -1 = Decreasing constraint (As feature goes up, prediction goes down)
    # 0 = No constraint
    
    # Map constraints to the feature list order:
    # 'discounted_price': -1 (Price UP -> Demand DOWN)
    # 'actual_price': 0 (Neutral)
    # 'discount_percentage': 1 (Discount UP -> Demand UP)
    # 'rating': 1 (Rating UP -> Demand UP)
    # 'desc_len': 1 (Better Description -> Demand UP)
    # 'price_competitiveness': -1 (More Expensive vs Peers -> Demand DOWN)
    # 'review_sentiment': 1 (Better Sentiment -> Demand UP)
    # 'main_category_encoded': 0 (Neutral)
    
    constraints = "(-1, 0, 1, 1, 1, -1, 1, 0)"
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=6,
        monotone_constraints=constraints  # <--- FORCING ECONOMIC LOGIC
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
fig.patch.set_alpha(0.0) # Transparent figure background

# Helper to style axes
def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor('#0f172a') # Dark inner background (optional, or transparent)
    ax.patch.set_alpha(0.0)      # Make inner plot transparent
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#94a3b8')
    ax.spines['left'].set_color('#94a3b8')
    ax.tick_params(axis='x', colors='#cbd5e1')
    ax.tick_params(axis='y', colors='#cbd5e1')
    ax.yaxis.label.set_color('#e2e8f0')
    ax.xaxis.label.set_color('#e2e8f0')
    ax.title.set_color('#f8fafc')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.1, color='#cbd5e1', linestyle='--')
    return ax

# Demand Curve
ax1 = style_ax(ax1, "Price vs. Demand", "Price (₹)", "Predicted Sales Volume")
ax1.plot(prices, demands, color='#22c55e', linewidth=2.5) # Green
ax1.axvline(new_price, color='#ef4444', linestyle='--', label='Selected Price', linewidth=1.5) # Red
ax1.axvline(current_price, color='#3b82f6', linestyle='--', label='Current Price', linewidth=1.5) # Blue
ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')

# Revenue Curve
ax2 = style_ax(ax2, "Price vs. Revenue", "Price (₹)", "Projected Revenue (₹)")
ax2.plot(prices, revenues, color='#a855f7', linewidth=2.5) # Purple
ax2.axvline(new_price, color='#ef4444', linestyle='--', label='Selected Price', linewidth=1.5)
ax2.axvline(current_price, color='#3b82f6', linestyle='--', label='Current Price', linewidth=1.5)
ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')

plt.tight_layout()
st.pyplot(fig, transparent=True)

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