# (Business-driven features — interview gold)

import pandas as pd

def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)

    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 12, 36, 72],
        labels=["New", "Mid", "Long"]
    )

    df["high_value_customer"] = (df["monthly_charges"] > 90).astype(int)

    df["support_intensity"] = pd.cut(
        df["support_tickets_last_3m"],
        bins=[-1, 1, 3, 10],
        labels=["Low", "Medium", "High"]
    )

    df = pd.get_dummies(
        df,
        columns=["tenure_bucket", "support_intensity"],
        drop_first=True
    )

    df.to_csv(output_path, index=False)
    print("✅ Feature engineering completed.")

if __name__ == "__main__":
    engineer_features(
        "data/processed/cleaned_churn_data.csv",
        "data/processed/feature_engineered_churn.csv"
    )