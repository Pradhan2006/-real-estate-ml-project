import os
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    """Loads the California Housing dataset."""
    print("Loading data...")
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    return X, y, california.feature_names, california.target_names

def train_model():
    """Trains a Random Forest Regressor and saves it."""
    X, y, feature_names, _ = load_data()
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model (Random Forest)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    print("Saving model...")
    os.makedirs('models', exist_ok=True)
    # Save the model and feature names together for the Streamlit app
    model_data = {
        'model': model,
        'features': feature_names
    }
    joblib.dump(model_data, 'models/model_data.joblib')
    print("Model saved to models/model_data.joblib")

if __name__ == "__main__":
    train_model()
