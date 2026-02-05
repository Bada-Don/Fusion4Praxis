import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob

app = FastAPI(title="Retail Pricing Optimization API")

# Allow requests from Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
MODEL_PATH = "api/model.pkl"
LE_PATH = "api/label_encoder.pkl"
DATA_PATH = "data/Pricing_dataset.csv"

# --- Global Variables ---
model = None
label_encoder = None

class PredictRequest(BaseModel):
    product_id: str = "test-item"
    price: float
    desc_len: int = 100
    sentiment: float = 0.5
    category: str = "Others"
    actual_price: float = 100.0 # Default fallback
    rating: float = 4.0 # Default fallback

class PredictResponse(BaseModel):
    predicted_demand: float
    projected_revenue: float
    price_elasticity: float
    violations: int = 0

# --- Helper Functions (Ported from app.py) ---

def clean_currency(x):
    if isinstance(x, str):
        x = x.replace('₹', '').replace(',', '').strip()
    return pd.to_numeric(x, errors='coerce')

def get_sentiment(text):
    if pd.isna(text):
        return 0
    return TextBlob(str(text)).sentiment.polarity

def load_and_prepare_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    # Cleaning Money Columns
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
    df.drop('cat_split', axis=1, inplace=True)
    
    # Visibility Metrics (Content Quality)
    df['desc_len'] = df['about_product'].astype(str).apply(len)
    df['name_len'] = df['product_name'].astype(str).apply(len)
    
    # Relative Pricing (Context)
    df['category_avg_price'] = df.groupby('sub_category')['discounted_price'].transform('mean')
    df['price_competitiveness'] = df['discounted_price'] / df['category_avg_price']
    
    # Sentiment Analysis
    df['review_sentiment'] = df['review_content'].apply(get_sentiment)
    
    # Target Transformation
    df['log_demand'] = np.log1p(df['rating_count'])
    
    return df

def train_model_logic(df):
    print("Training new model...")
    # Encode Categoricals
    le = LabelEncoder()
    df['main_category_encoded'] = le.fit_transform(df['main_category'])
    
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
    
    X = df[features]
    y = df['log_demand']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Monotonic Constraints
    constraints = "(-1, 0, 1, 1, 1, -1, 1, 0)"
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=6,
        monotone_constraints=constraints
    )
    
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model, le

# --- Lifecycle Logic ---

@app.on_event("startup")
def startup_event():
    global model, label_encoder
    
    # Check if model exists
    if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
        print(f"Loading saved model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(LE_PATH)
        print("Model loaded successfully.")
    else:
        print("Saved model not found. Starting training process...")
        try:
            df = load_and_prepare_data()
            model, label_encoder = train_model_logic(df)
            
            # Save to disk
            joblib.dump(model, MODEL_PATH)
            joblib.dump(label_encoder, LE_PATH)
            print(f"Model saved to {MODEL_PATH}")
        except Exception as e:
            print(f"Error during training: {e}")
            # Fallback to mock if training fails (optional, but good for stability)
            model = "MOCK" 

@app.get("/")
def read_root():
    status = "Active (Real Model)" if model != "MOCK" and model is not None else "Active (Mock/Loading)"
    return {"status": status, "message": "Pricing Optimization Model Ready"}

@app.post("/predict", response_model=PredictResponse)
def predict_demand(request: PredictRequest):
    global model, label_encoder
    
    # Fallback to simple logic if model isn't ready or failed
    if model == "MOCK" or model is None:
        # Simple heuristic fallback (copied from previous mock)
        demand = max(0, 1000 - 5.0 * request.price + min(request.desc_len, 500) * 0.5)
        return PredictResponse(
            predicted_demand=demand,
            projected_revenue=demand * request.price,
            price_elasticity=-1.5,
            violations=0
        )

    try:
        # Prepare Input
        # Note: We need to recreate 'price_competitiveness' and 'discount_percentage'
        # To do this accurately, we really need the 'category_avg_price' for the subcategory.
        # Since we don't have the full DF here efficiently, we can approximate or use a passed value.
        # For this Hackathon implementation, we'll approximate competitiveness using the price itself vs a standard.
        # Or better, we trust the features we can compute.
        
        # Approximate: Assume average price is mostly stable around the actual_price or close to it.
        # Let's derive it from the passed actual_price (if realistic).
        
        category_encoded = 0
        if label_encoder:
            try:
                category_encoded = label_encoder.transform([request.category])[0]
            except:
                category_encoded = 0 # Default/Other
        
        # Derived features
        discount_percentage = ((request.actual_price - request.price) / request.actual_price) * 100
        # If we don't have data, assume competitiveness is 1.0 (average) if price == actual_price
        # A simple proxy: estimate category avg price as (actual_price * 0.8)
        est_avg_price = request.actual_price * 0.8 
        price_competitiveness = request.price / est_avg_price
        
        input_features = pd.DataFrame([{
            'discounted_price': request.price,
            'actual_price': request.actual_price,
            'discount_percentage': discount_percentage,
            'rating': request.rating,
            'desc_len': request.desc_len,
            'price_competitiveness': price_competitiveness,
            'review_sentiment': request.sentiment,
            'main_category_encoded': category_encoded
        }])
        
        log_demand = model.predict(input_features)[0]
        demand = np.expm1(log_demand)
        
        # Ensure non-negative
        demand = max(0, demand)
        revenue = demand * request.price
        
        return PredictResponse(
            predicted_demand=round(float(demand), 2),
            projected_revenue=round(float(revenue), 2),
            price_elasticity=-1.81, # Static or could be computed by probing model
            violations=0
        )
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
