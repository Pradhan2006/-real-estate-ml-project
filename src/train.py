import os
import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_model():
    print("Loading dataset...")

    data = fetch_california_housing(as_frame=True)
    df = data.frame

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating...")
    preds = model.predict(X_test)

    print("MSE:", mean_squared_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))

    print("Saving model...")

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": X.columns
        },
        "models/model_data.joblib"
    )

    print("Model saved successfully ✔")


if __name__ == "__main__":
    train_model()