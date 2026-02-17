import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------
# Page Configuration
# --------------------------------------------
st.set_page_config(
    page_title="Sales Forecast App",
    layout="wide"
)

st.title("📊 Sales Forecasting App")
st.markdown("### Powered by Tuned LightGBM Model")

# --------------------------------------------
# Safe Model Path (Production Ready)
# --------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_lightgbm_sales_model.pkl")

# --------------------------------------------
# Cached Model Loader
# --------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

# --------------------------------------------
# Extract Feature Names from Pipeline
# --------------------------------------------
preprocessor = model.named_steps["preprocessor"]

numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]

# --------------------------------------------
# Sidebar Inputs
# --------------------------------------------
st.sidebar.header("Enter Feature Values")

input_data = {}

# Numeric Inputs
for col in numeric_features:
    input_data[col] = st.sidebar.number_input(
        label=col,
        value=0.0,
        format="%.4f"
    )

# Categorical Inputs (Safe Dropdown from Encoder)
cat_transformer = preprocessor.transformers_[1][1]
encoder = cat_transformer.named_steps["onehot"]

for i, col in enumerate(categorical_features):
    categories = encoder.categories_[i]
    input_data[col] = st.sidebar.selectbox(
        label=col,
        options=categories
    )

# Create DataFrame
input_df = pd.DataFrame([input_data])

# --------------------------------------------
# Prediction Section
# --------------------------------------------
st.markdown("### Input Preview")
st.dataframe(input_df)

if st.button("🚀 Predict Sales"):

    try:
        prediction = model.predict(input_df)[0]

        st.success("Prediction Complete!")
        st.metric(
            label="Predicted Sales",
            value=f"{prediction:,.2f}"
        )

    except Exception as e:
        st.error("Prediction failed. Check input values.")
        st.exception(e)
