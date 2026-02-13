# (Business output: risk + revenue impact)

import pandas as pd
import numpy as np
import joblib

def score_customers(data_path, output_path):
    df = pd.read_csv(data_path)

    model = joblib.load("models/rf_churn_model_jupyter.pkl")

    X = df.drop(columns=["customer_id", "churn"])
    df["churn_probability"] = np.round((model.predict_proba(X)[:, 1]), 2)

    df["risk_segment"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )

    df["revenue_at_risk"] = np.round((df["monthly_charges"] * df["churn_probability"]), 2)

    df.to_csv(output_path, index=False)
    print("✅ Customer scoring completed.")

if __name__ == "__main__":
    score_customers(
        "data/processed/feature_engineered_churn_jupyter.csv",
        "reports/rf_churn_risk_scored_customers_jupyter.csv"
    )