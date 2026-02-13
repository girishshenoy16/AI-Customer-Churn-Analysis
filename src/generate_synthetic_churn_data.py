# (Data generation — already interview-defensible)

import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000

customer_id = [f"CUST_{i}" for i in range(1, N + 1)]

gender = np.random.choice(["Male", "Female"], size=N)
senior_citizen = np.random.choice([0, 1], size=N, p=[0.85, 0.15])

tenure_months = np.random.gamma(shape=2, scale=12, size=N).astype(int)
tenure_months = np.clip(tenure_months, 1, 72)

contract_type = np.random.choice(
    ["Month-to-Month", "One Year", "Two Year"],
    size=N,
    p=[0.55, 0.25, 0.20]
)

monthly_charges = np.round(np.random.normal(70, 25, N), 2)
monthly_charges = np.clip(monthly_charges, 20, 150)

total_charges = np.round(monthly_charges * tenure_months, 2)

payment_method = np.random.choice(
    ["Credit Card", "Debit Card", "UPI", "Net Banking"],
    size=N
)

avg_monthly_usage = np.round(np.random.normal(300, 100, N), 1)
avg_monthly_usage = np.clip(avg_monthly_usage, 50, 600)

usage_trend = np.random.choice(
    ["Increasing", "Stable", "Decreasing"],
    size=N,
    p=[0.25, 0.45, 0.30]
)

support_tickets_last_3m = np.clip(
    np.random.poisson(1.2, N), 0, 10
)

churn_prob = (
    0.35 * (tenure_months < 12) +
    0.25 * (contract_type == "Month-to-Month") +
    0.15 * (monthly_charges > 90) +
    0.15 * (usage_trend == "Decreasing") +
    0.10 * (support_tickets_last_3m >= 3)
)

churn_prob = np.clip(churn_prob, 0, 0.85)
churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    "customer_id": customer_id,
    "gender": gender,
    "senior_citizen": senior_citizen,
    "tenure_months": tenure_months,
    "contract_type": contract_type,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "payment_method": payment_method,
    "avg_monthly_usage": avg_monthly_usage,
    "usage_trend": usage_trend,
    "support_tickets_last_3m": support_tickets_last_3m,
    "churn": churn
})

df.to_csv("data/raw/synthetic_customer_churn.csv", index=False)
print("✅ Synthetic churn data generated.")