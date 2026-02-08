# preprocess.py — Run ONCE locally, commit output files
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

df = pd.read_csv("data/Pricing_dataset.csv")

def clean_currency(x):
    if isinstance(x, str):
        x = x.replace('₹', '').replace(',', '').strip()
    return pd.to_numeric(x, errors='coerce')

df['discounted_price'] = df['discounted_price'].apply(clean_currency)
df['actual_price'] = df['actual_price'].apply(clean_currency)
df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['discounted_price', 'actual_price', 'rating_count', 'rating'], inplace=True)

cat_split = df['category'].str.split('|')
df['main_category'] = cat_split.apply(lambda x: x[0] if isinstance(x, list) else "Other")
df['sub_category'] = cat_split.apply(lambda x: x[-1] if isinstance(x, list) else "Other")

df['desc_len'] = df['about_product'].astype(str).apply(len)
df['name_len'] = df['product_name'].astype(str).apply(len)
df['category_avg_price'] = df.groupby('sub_category')['discounted_price'].transform('mean')
df['price_competitiveness'] = df['discounted_price'] / df['category_avg_price']

def get_sentiment(text):
    if pd.isna(text):
        return 0
    return TextBlob(str(text)).sentiment.polarity

df['review_sentiment'] = df['review_content'].apply(get_sentiment)
df['log_demand'] = np.log1p(df['rating_count'])

# Save processed data
df.to_csv("data/processed_data.csv", index=False)

# Train and save model
le = LabelEncoder()
df['main_category_encoded'] = le.fit_transform(df['main_category'])

features = [
    'discounted_price', 'actual_price', 'discount_percentage', 'rating',
    'desc_len', 'price_competitiveness', 'review_sentiment', 'main_category_encoded'
]
X = df[features]
y = df['log_demand']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=500,
    learning_rate=0.05, max_depth=6,
    monotone_constraints="(-1, 0, 1, 1, 1, -1, 1, 0)"
)
model.fit(X_train, y_train)

model.save_model("data/xgb_model.json")
joblib.dump(le, "data/label_encoder.joblib")
print("Done! Commit data/processed_data.csv, data/xgb_model.json, data/label_encoder.joblib")
