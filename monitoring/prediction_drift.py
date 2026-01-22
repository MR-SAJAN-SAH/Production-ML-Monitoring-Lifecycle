import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_v1.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "models", "feature_names.pkl")
PROD_PATH = os.path.join(PROJECT_ROOT, "data", "production", "production_data.csv")


def detect_prediction_drift(threshold=0.05):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    df = pd.read_csv(PROD_PATH)

    df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_names]
    X_scaled = scaler.transform(X)

    preds = model.predict_proba(X_scaled)[:, 1]

    baseline = np.random.normal(0.3, 0.05, size=len(preds))  # simulated baseline

    stat, p_value = ks_2samp(baseline, preds)

    drift_detected = p_value < threshold

    print("Prediction Drift Report")
    print(f"p-value: {p_value:.4f}")
    print("Drift Detected" if drift_detected else "No Drift")

    return drift_detected


if __name__ == "__main__":
    detect_prediction_drift()
