import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Customer Churn Risk Analyzer",
    layout="wide"
)

st.markdown(
    """
    <style>
        .block-container {
            text-align: center;
        }
        div[data-testid="metric-container"] {
            text-align: center;
        }
        table {
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load model and data
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/rf_churn_model_jupyter.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("reports/rf_churn_risk_scored_customers_jupyter.csv")

model = load_model()
df = load_data()

# -----------------------------
# Title
# -----------------------------
st.title("📉 AI-Driven Customer Churn Risk Analyzer")

st.markdown(
    "Predict churn risk, identify high-value customers, and estimate revenue impact."
)

# -----------------------------
# KPI Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
churn_rate = df["churn"].mean() * 100
high_risk_customers = (df["risk_segment"] == "High").sum()
revenue_at_risk = df["revenue_at_risk"].sum()

with col1:
    st.markdown("<p style='text-align:center;'>Total Customers</p>", unsafe_allow_html=True)
    st.metric(label="", value=f"{total_customers:,}")

with col2:
    st.markdown("<p style='text-align:center;'>Churn Rate (%)</p>", unsafe_allow_html=True)
    st.metric(label="", value=f"{churn_rate:.2f}")

with col3:
    st.markdown("<p style='text-align:center;'>High-Risk Customers</p>", unsafe_allow_html=True)
    st.metric(label="", value=f"{high_risk_customers:,}")

with col4:
    st.markdown("<p style='text-align:center;'>Revenue at Risk (₹)</p>", unsafe_allow_html=True)
    st.metric(label="", value=f"{revenue_at_risk:,.0f}")

st.divider()

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("🔎 Filter Customers")

risk_filter = st.sidebar.multiselect(
    "Select Risk Segment",
    options=df["risk_segment"].unique(),
    default=df["risk_segment"].unique()
)

filtered_df = df[df["risk_segment"].isin(risk_filter)]

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("📊 Churn Risk Distribution")

risk_counts = filtered_df["risk_segment"].value_counts()

fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
risk_counts.plot(kind="bar", ax=ax)

ax.set_xlabel("Risk Segment")
ax.set_ylabel("Number of Customers")
ax.set_title("Customer Distribution by Churn Risk")
ax.tick_params(axis="x", rotation=0, labelsize=9)
ax.tick_params(axis="y", labelsize=9)

left, center, right = st.columns([1, 2, 1])

with center:
    st.pyplot(fig, use_container_width=False)

st.divider()

st.subheader("💰 Revenue at Risk by Segment")
revenue_by_risk = (
    filtered_df
    .groupby("risk_segment")["revenue_at_risk"]
    .sum()
)

fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
revenue_by_risk.plot(kind="bar", ax=ax)

ax.set_xlabel("Risk Segment")
ax.set_ylabel("Revenue at Risk")
ax.set_title("Revenue at Risk by Churn Segment")
ax.tick_params(axis="x", rotation=0, labelsize=9)
ax.tick_params(axis="y", labelsize=9)

left, center, right = st.columns([1, 2, 1])

with center:
    st.pyplot(fig, use_container_width=False)

st.divider()

# -----------------------------
# Customer Table
# -----------------------------
st.subheader("👥 Customer-Level Churn Scoring")

st.dataframe(
    filtered_df[[
        "customer_id",
        "monthly_charges",
        "tenure_months",
        "churn_probability",
        "risk_segment",
        "revenue_at_risk"
    ]].sort_values("churn_probability", ascending=False),
    use_container_width=True
)

st.divider()