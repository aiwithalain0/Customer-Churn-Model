# 📊 CustomerIQ — Enterprise Customer Churn Prediction & Explainable AI (XAI)

An enterprise-grade, full-stack Machine Learning web application for predicting customer churn risk and delivering transparent factor attribution. Built with **FastAPI**, **XGBoost**, **SHAP (SHapley Additive exPlanations)**, and a custom **Dashboard UI** replicating modern analytics software design.

----- 
    
## 🌟 Key Features

- **Full-Stack Dashboard Architecture**:
  - **FastAPI Backend**: Loads model artifacts once at startup, serves RESTful analytics endpoints, dynamic cohort slicing, and on-the-fly SHAP force plots.
  - **Modern Dashboard Frontend**: Soft gray canvas (`#f8fafc`), crisp white card elevation, custom typography (Inter), responsive multi-panel layout, and interactive Chart.js visualizations.
- **Trained XGBoost Classifier**: Powered by pre-trained artifacts (`final_model.pkl`, `scaler.pkl`, `model_columns.pkl`, `shap_explainer.pkl`).
- **Real-Time Data & Computations**:
  - **4 KPI Cards**: Real counts for Total Customers (7,043), Model-Predicted At-Risk Accounts (3,063), Ground-Truth Churn Rate (26.5%), and Computed Expected Retention (56.7%).
  - **Churn Prediction Overview**: Line trend chart mapping churned vs. retained accounts across tenure cohorts (Nov – Apr proxy timeline).
  - **Churn Risk Distribution**: Donut chart displaying customer segmentation into High (>60%), Medium (35%–60%), and Low (<35%) risk tiers.
  - **Top Churn Factors**: Ranked feature impacts computed from mean absolute SHAP values across the entire dataset.
  - **Recent Predictions Table**: Displays real customer records with probability scores, risk badges, and interactive **"View"** triggers.
  - **Individual SHAP Modal**: Detailed pop-up rendering instant **SHAP force plots** and key driver directionality for any selected account.
  - **Dynamic Insights & Recommendations**: Text recommendations generated directly from top SHAP feature rankings.
  - **Export Report**: Instant CSV download of all at-risk accounts.

---

## 📁 Repository Structure

```text
Customer_Churn_Model/
├── server.py                  # FastAPI server & inference / SHAP endpoints
├── static/
│   ├── index.html             # Multi-panel dashboard UI (matching reference design)
│   ├── styles.css             # Custom design system, typography, and card styling
│   └── app.js                 # Dynamic API integration, Chart.js, and SHAP modal viewer
├── app.py                     # Standalone Gradio web application
├── CustomerChurnModel.ipynb    # Full ML Development Notebook (EDA, Modeling, SHAP)
├── Telco-Customer-Churn.csv    # Raw IBM Telco Customer Churn Dataset
├── final_model.pkl            # Trained & Tuned XGBoost Classifier
├── scaler.pkl                 # Fitted StandardScaler for Numeric Features
├── model_columns.pkl          # Exact One-Hot Encoded Training Feature Order
├── shap_explainer.pkl         # Fitted SHAP TreeExplainer
├── requirements.txt           # Pinned Python Dependencies
├── .gitignore                 # Standard Python / ML Ignore Rules
└── README.md                  # Comprehensive Documentation & Architecture Guide
```

---

## 🛠️ Installation & Setup

### 1. Set Up Environment & Dependencies
```bash
# Clone or navigate to repository directory
cd Customer_Churn_Model

# Install requirements
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Launch the CustomerIQ Web App (FastAPI + Dashboard UI):
```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```
Open your browser at:
👉 **`http://127.0.0.1:8000/`**

---

## 📊 Live Endpoints

- `GET /` — Serves the CustomerIQ Dashboard Single-Page Application.
- `GET /api/dashboard-stats?segment={segment}` — Computes KPIs, line trends, risk donut distribution, top SHAP factors, and recommendations.
- `GET /api/customers?limit=6&search={query}` — Returns paginated customer predictions with risk levels.
- `GET /api/customer/{customer_id}/shap` — Generates on-the-fly individual SHAP force plot and feature driver list.
- `POST /api/predict` — Simulates live churn probabilities and returns custom SHAP plots for what-if scenarios.
- `GET /api/export-report` — Downloads a CSV summary report of high and medium risk accounts.

---

## 📄 License
This project is licensed under the MIT License.
