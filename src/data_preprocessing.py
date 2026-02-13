# (Cleaning + encoding — production logic)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df.drop_duplicates(inplace=True)

    categorical_cols = [
        "gender",
        "contract_type",
        "payment_method",
        "usage_trend"
    ]

    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    df.to_csv(output_path, index=False)
    print("✅ Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data(
        "data/raw/synthetic_customer_churn.csv",
        "data/processed/cleaned_churn_data.csv"
    )