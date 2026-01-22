import os
import pandas as pd
from scipy.stats import ks_2samp


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REF_PATH = os.path.join(PROJECT_ROOT, "data", "reference", "reference_data.csv")
PROD_PATH = os.path.join(PROJECT_ROOT, "data", "production", "production_data.csv")


def detect_data_drift(threshold=0.05):
    ref = pd.read_csv(REF_PATH)
    prod = pd.read_csv(PROD_PATH)

    drift_report = {}

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    for col in numeric_cols:
        stat, p_value = ks_2samp(ref[col], prod[col])
        drift_report[col] = p_value < threshold

    print("📊 Data Drift Report")
    for feature, drifted in drift_report.items():
        status = "DRIFT DETECTED" if drifted else "No drift"
        print(f"{feature}: {status}")

    return any(drift_report.values())


if __name__ == "__main__":
    drift_found = detect_data_drift()
    print("\nOverall Drift:", drift_found)
