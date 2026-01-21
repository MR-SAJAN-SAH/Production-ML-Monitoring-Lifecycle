import os
import yaml
import joblib
import pandas as pd
from fastapi import FastAPI

from schema import CustomerInput, PredictionResponse


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(PROJECT_ROOT, "config", "config.yaml")) as f:
    config = yaml.safe_load(f)

MODEL_PATH = os.path.join(PROJECT_ROOT, config["paths"]["models"]["model_active"])
SCALER_PATH = os.path.join(PROJECT_ROOT, config["paths"]["models"]["scaler"])
FEATURES_PATH = os.path.join(PROJECT_ROOT, config["paths"]["models"]["features"])

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

app = FastAPI(title="Production ML Monitoring API")


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerInput):
    input_df = pd.DataFrame([data.dict()])

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]
    scaled_input = scaler.transform(input_df)

    probability = model.predict_proba(scaled_input)[0][1]

    return PredictionResponse(churn_probability=probability)
