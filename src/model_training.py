# (Train + save model — clean ML pipeline)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/churn_model.pkl")
    print("✅ Model trained and saved.")

    return X_test, y_test

if __name__ == "__main__":
    train_model("data/processed/feature_engineered_churn.csv")