import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import subprocess

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ================= CSS =================
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .price-text {
        font-size: 32px;
        font-weight: bold;
        color: #28a745;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ================= MODEL PATH =================
MODEL_PATH = "models/model_data.joblib"

# ================= LOAD MODEL (SAFE) =================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Training model for first run... please wait ⏳")
        subprocess.run(["python3", "src/train.py"])

    if not os.path.exists(MODEL_PATH):
        st.error("Model not created. Check train.py logs.")
        st.stop()

    return joblib.load(MODEL_PATH)

# ================= MAIN APP =================
def main():
    st.markdown("<h1 class='main-header'>🏠 California Real Estate Price Predictor</h1>", unsafe_allow_html=True)

    model_data = load_model()
    model = model_data["model"]
    features = model_data["features"]

    st.write("Adjust values below to predict house price:")

    col1, col2 = st.columns(2)

    input_data = {}

    with col1:
        st.subheader("Property")
        input_data["MedInc"] = st.slider("Median Income", 0.5, 15.0, 3.5)
        input_data["HouseAge"] = st.slider("House Age", 1, 100, 25)
        input_data["AveRooms"] = st.slider("Avg Rooms", 1.0, 20.0, 5.0)
        input_data["AveBedrms"] = st.slider("Avg Bedrooms", 0.5, 10.0, 1.0)

    with col2:
        st.subheader("Location")
        input_data["Population"] = st.slider("Population", 10, 10000, 1000)
        input_data["AveOccup"] = st.slider("Avg Occupancy", 1.0, 10.0, 3.0)
        input_data["Latitude"] = st.slider("Latitude", 32.0, 42.0, 35.0)
        input_data["Longitude"] = st.slider("Longitude", -125.0, -114.0, -120.0)

    st.markdown("---")

    if st.button("Predict Price", use_container_width=True):
        df = pd.DataFrame([input_data])
        df = df[features]

        prediction = model.predict(df)[0]
        price = prediction * 100000

        st.markdown(f"""
        <div class="prediction-box">
            <h3 style='text-align:center;'>Estimated House Value</h3>
            <div class="price-text">${price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()


# ================= RUN =================
if __name__ == "__main__":
    main()