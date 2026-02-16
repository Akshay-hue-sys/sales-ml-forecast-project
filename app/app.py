import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Sales Forecast App", layout="wide")

st.title("📊 Sales Forecasting App")
st.write("Prediction powered by Tuned LightGBM Model")

# ---------------------------------
# Load Model
# ---------------------------------
MODEL_PATH = os.path.join("..", "models", "final_lightgbm_sales_model.pkl")

model = joblib.load(MODEL_PATH)

# ---------------------------------
# Feature Inputs
# ---------------------------------
st.sidebar.header("Enter Feature Values")

numeric_features = model.named_steps["preprocessor"].transformers_[0][2]
categorical_features = model.named_steps["preprocessor"].transformers_[1][2]

input_data = {}

for col in numeric_features:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

for col in categorical_features:
    input_data[col] = st.sidebar.text_input(col, "")

input_df = pd.DataFrame([input_data])

# ---------------------------------
# Prediction Button
# ---------------------------------
if st.button("Predict Sales"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")
