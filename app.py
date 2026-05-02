import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config for aesthetics
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS for aesthetics
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

@st.cache_resource
def load_model_data():
    try:
        return joblib.load("models/model_data.joblib")
    except FileNotFoundError:
        return None

def main():
    st.markdown("<h1 class='main-header'>🏠 California Real Estate Price Predictor</h1>", unsafe_allow_html=True)
    
    model_data = load_model_data()
    
    if model_data is None:
        st.error("Model file not found! Please run `python src/train.py` first to train the model.")
        return
        
    model = model_data['model']
    features = model_data['features']
    
    st.write("This application predicts the median house value in a California block group based on various features. Please adjust the parameters below.")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    # Distribute features across columns nicely
    # 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'
    
    with col1:
        st.subheader("Property Characteristics")
        input_data['MedInc'] = st.slider("Median Income (in $10k)", min_value=0.5, max_value=15.0, value=3.5, step=0.1, help="Median income in block group")
        input_data['HouseAge'] = st.slider("House Age (years)", min_value=1, max_value=100, value=25, step=1, help="Median house age in block group")
        input_data['AveRooms'] = st.slider("Average Rooms", min_value=1.0, max_value=20.0, value=5.0, step=0.1, help="Average number of rooms per household")
        input_data['AveBedrms'] = st.slider("Average Bedrooms", min_value=0.5, max_value=10.0, value=1.0, step=0.1, help="Average number of bedrooms per household")

    with col2:
        st.subheader("Location & Demographics")
        input_data['Population'] = st.slider("Population", min_value=10, max_value=10000, value=1000, step=10, help="Block group population")
        input_data['AveOccup'] = st.slider("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1, help="Average number of household members")
        input_data['Latitude'] = st.slider("Latitude", min_value=32.0, max_value=42.0, value=35.0, step=0.1)
        input_data['Longitude'] = st.slider("Longitude", min_value=-125.0, max_value=-114.0, value=-120.0, step=0.1)
        
    st.markdown("---")
    
    # Predict Button
    if st.button("Predict Price", type="primary", use_container_width=True):
        # Format input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure order matches training features
        input_df = input_df[features]
        
        # Predict (model returns target in hundreds of thousands of dollars)
        prediction = model.predict(input_df)[0]
        
        # Format prediction ($100k -> actual $)
        actual_price = prediction * 100000
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3 style='text-align: center; color: #555;'>Estimated Median House Value</h3>
            <div class="price-text">${actual_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()

if __name__ == "__main__":
    main()
import os
import joblib
import streamlit as st
import subprocess

MODEL_PATH = "models/model_data.joblib"

# If model doesn't exist, train it
if not os.path.exists(MODEL_PATH):
    st.info("Training model for first run... please wait ⏳")
    subprocess.run(["python3", "src/train.py"])

# Load model
model = joblib.load(MODEL_PATH)