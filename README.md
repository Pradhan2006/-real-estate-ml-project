# Real Estate Price Prediction

This is an end-to-end Machine Learning project that predicts the median house value for California districts based on demographic and property data. It uses `scikit-learn` for training a Random Forest Regression model and `Streamlit` for serving the model via a clean web interface.

## Project Structure

- `src/train.py`: Script to download data, train the model, and save it.
- `app.py`: Streamlit web application to serve predictions.
- `models/`: Directory where the trained model (`model_data.joblib`) is saved.
- `requirements.txt`: Python dependencies.

## Setup Instructions

1. **Navigate to the project directory:**
   ```bash
   cd real-estate-price-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model:**
   Run the training script to fetch data and generate the `.joblib` model file.
   ```bash
   python src/train.py
   ```

5. **Run the Application:**
   Start the Streamlit server.
   ```bash
   streamlit run app.py
   ```

## Dataset
The dataset used is the California Housing dataset, originally published in *Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 33 (1997) 291-297*.
