import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROD_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "production", "production_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def retrain_pipeline():
    df = pd.read_csv(PROD_DATA_PATH)

    df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = pd.get_dummies(df, drop_first=True)

    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names + ["Churn"]]

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    joblib.dump(model, os.path.join(MODEL_DIR, "model_v2.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("🔁 Model retrained and promoted to v2")


if __name__ == "__main__":
    retrain_pipeline()
