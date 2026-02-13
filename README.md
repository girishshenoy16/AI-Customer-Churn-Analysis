# 📉 AI-Driven Customer Churn Prediction & Revenue Impact Analyzer For Subscription Businesses

## 📌 Project Overview

Customer churn is a critical challenge for subscription-based businesses such as SaaS, telecom, and fintech companies.
This project implements an **end-to-end AI-driven churn analysis system** that predicts customer churn, quantifies **revenue at risk**, and presents insights through an **interactive Streamlit dashboard**.

The project is intentionally designed to reflect **real AI Analyst workflows**, where **Jupyter notebooks are used for analysis and model development**, and reusable scripts represent how the same logic can be automated in production.

---

## 🎯 Business Problem

Most subscription businesses detect churn **after customers have already left**, resulting in lost revenue and missed retention opportunities.

**Objective:**
Identify customers who are likely to churn *in advance* and estimate the potential revenue impact so that business teams can take proactive retention actions.

---

## 🧠 Solution Approach

The solution follows a complete analytics and machine learning lifecycle:

1. Generate business-realistic synthetic customer data
2. Perform exploratory data analysis (EDA) to identify churn drivers
3. Clean and engineer features based on business insights
4. Train and evaluate churn prediction models in Jupyter notebooks
5. Persist trained models for reuse
6. Generate churn risk and revenue impact reports
7. Visualize results using a Streamlit dashboard

---

## 🧪 Model Development Workflow (IMPORTANT)

Model development and feature engineering were performed **primarily in Jupyter notebooks**, which serve as the **source of truth** for analytical decisions and final outputs.

* Data cleaning and feature engineering were finalized in notebooks
* Logistic Regression and Random Forest models were trained and evaluated in notebooks
* Final trained models were saved using `joblib`
* Reports and dashboards are based on these notebook-trained models

The `src/` directory contains **modular Python scripts that mirror the same workflow**, demonstrating how the analysis could be automated or productionized in a real-world setting.

---

## 📊 Key Business Metrics

* Churn Probability (per customer)
* Risk Segmentation (Low / Medium / High)
* Revenue at Risk
* Overall Churn Rate
* High-Risk Customer Count

---

## 🏗️ Project Architecture

```
Synthetic Customer Data
        ↓
Exploratory Data Analysis (Notebook)
        ↓
Data Cleaning & Feature Engineering (Notebook)
        ↓
Model Training & Evaluation (Notebook)
        ↓
Saved Models (joblib)
        ↓
Churn Risk & Revenue Reports
        ↓
Streamlit Dashboard
```

---

## 🗂️ Project Structure

```
ai-customer-churn/
│
├── data/
│   ├── raw/                 # Synthetic raw data
│   └── processed/           # Cleaned & feature-engineered data (from notebooks)
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_and_evaluation.ipynb
│
├── src/
│   ├── generate_synthetic_churn_data.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── predict_and_score.py
│
├── models/
│   ├── rf_churn_model_jupyter.pkl       # Final model trained in notebooks
│   ├── lr_churn_model_jupyter.pkl             
│   └── churn_model.pkl  
│
├── reports/
│   ├── rf_churn_risk_scored_customers_jupyter.csv       # Final model trained in notebooks
│   ├── lr_churn_risk_scored_customers_jupyter.csv       
│   └── churn_risk_scored_customers.csv
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🧾 Data Description

The dataset represents customer behavior in a subscription business and includes:

* Customer tenure
* Monthly charges
* Contract type
* Usage trends
* Support interactions
* Churn label (target variable)

> ⚠️ The data is **synthetic but business-realistic**, designed to reflect real churn patterns commonly seen in industry.

---

## 🤖 Machine Learning Models

### Baseline Model

* **Logistic Regression**
* Used for interpretability and performance benchmarking

### Final Model

* **Random Forest Classifier**
* Selected due to:

  * Strong ROC-AUC performance
  * Ability to model non-linear behavior
  * Alignment with business intuition

### Evaluation Metrics

* ROC-AUC
* Precision / Recall
* F1-score
* Business relevance of false negatives (missed churners)

---

## 📈 Results Summary

* The model demonstrates strong discrimination between churned and retained customers
* Key churn drivers identified:

  * Low customer tenure
  * Month-to-month contracts
  * Decreasing usage trends
  * High support interaction
  * Higher monthly charges
* Outputs are directly usable for retention and revenue planning

---

## 🖥️ Streamlit Dashboard

The Streamlit application provides:

* KPI cards (Churn Rate, Revenue at Risk, High-Risk Customers)
* Churn risk distribution
* Revenue at risk analysis
* Customer-level churn scoring table

---

## ⚙️ How to Run the Project

### Clone repository:

```
git clone https://github.com/girishshenoy16/AI-Customer-Churn-Analysis
cd AI-Customer-Churn
```
### 1️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
python.exe -m pip install --upgrade pip
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Generate Synthetic Dataset
```bash
python src/generate_synthetic_churn_data.py
```

### 4️⃣ (Optional) Re-run analysis

Open and run the notebooks in order:

1. `01_exploratory_data_analysis.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_modeling_and_evaluation.ipynb`

### 5️⃣ Run dashboard locally 

```bash
streamlit run streamlit_app.py
```

### Optional Steps after generating synthetic dataset

### 3️⃣ Data Cleaning And Preprocessing
```bash
python src/data_preprocessing.py
```

### 4️⃣ Feature Engineering
```bash
python src/feature_engineering.py
```

### 6️⃣ Model Training
```bash
python src/model_training.py
```

### 7️⃣ Model Evaluation
```bash
python src/model_evaluation.py
```

### 9️⃣ Predict And Score High Risked Customer
```bash
python src/predict_and_score.py
```

### 🔟 Launch dashboard locally 

```bash
streamlit run streamlit_app.py
```

---

## 🧠 Skills Demonstrated

* Business-driven exploratory data analysis
* Feature engineering based on real churn signals
* Supervised machine learning for classification
* Model evaluation using business-relevant metrics
* Model persistence and reuse
* Dashboarding and stakeholder-focused visualization
* Structuring projects for real-world workflows

---

## 🚀 Future Improvements

* Time-series churn modeling
* Real-time prediction APIs
* Automated model retraining
* A/B testing of retention strategies
* Integration with CRM systems

---

## 👤 Author

**Girish Shenoy**
Computer Science Student | Aspiring AI & Business Analyst

---

## ⭐ Final Note

This project emphasizes **clarity, business value, and execution quality**, reflecting how churn analysis is performed in real organizations rather than as an academic exercise.

---