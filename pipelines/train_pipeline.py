import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "telco_customer_churn_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def load_and_prepare_data(path):
    df = pd.read_csv(path)

    df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y


def train_pipeline():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_and_prepare_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

    joblib.dump(model, os.path.join(MODEL_DIR, "model_v1.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_names.pkl"))

    print(f"Initial model trained | ROC-AUC: {roc:.4f}")


if __name__ == "__main__":
    train_pipeline()
