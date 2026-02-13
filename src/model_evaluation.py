# (Metrics the way companies expect)

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    model = joblib.load("models/churn_model.pkl")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print("📊 Classification Report")
    print(classification_report(y, y_pred))
    print("ROC-AUC:", roc_auc_score(y, y_prob))

if __name__ == "__main__":
    evaluate_model("data/processed/feature_engineered_churn.csv")