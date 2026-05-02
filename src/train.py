import os
import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    print("Loading data...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    feature_names = X.columns
    return X, y, feature_names


def train_model():
    X, y, feature_names = load_data()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model (Random Forest)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    print("Saving model...")

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": feature_names
        },
        "models/model_data.joblib"
    )

    print("Model saved successfully ✔")


if __name__ == "__main__":
    train_model()