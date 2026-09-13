"""
Customer Churn Predictor - Modern Gradio Web Application
-------------------------------------------------------
Trained XGBoost Classifier with SHAP Explainability & Real-Time Risk Profiling.
Loads artifacts once at startup:
- final_model.pkl : Trained XGBoost classifier
- scaler.pkl : Fitted StandardScaler for numeric features
- model_columns.pkl : List of exact one-hot encoded columns in training order
- shap_explainer.pkl : Fitted SHAP TreeExplainer
- Telco-Customer-Churn.csv : Used to dynamically fetch exact category values and min/max ranges
"""

import io
import warnings
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import gradio as gr

# Suppress sklearn unpickle version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. LOAD ARTIFACTS AND DATASET AT STARTUP
# ==========================================
print("Loading model artifacts...")
FINAL_MODEL = joblib.load("final_model.pkl")
SCALER = joblib.load("scaler.pkl")
MODEL_COLUMNS = joblib.load("model_columns.pkl")
SHAP_EXPLAINER = joblib.load("shap_explainer.pkl")
print("Artifacts loaded successfully.")

# Read CSV to dynamically populate exact category options and numerical ranges
CSV_PATH = "Telco-Customer-Churn.csv"
try:
    df_raw = pd.read_csv(CSV_PATH)
except Exception:
    df_raw = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Extract unique choices dynamically from CSV
GENDER_CHOICES = sorted(df_raw["gender"].dropna().unique().tolist())
SENIOR_CHOICES = ["No (0)", "Yes (1)"]
PARTNER_CHOICES = sorted(df_raw["Partner"].dropna().unique().tolist())
DEPENDENTS_CHOICES = sorted(df_raw["Dependents"].dropna().unique().tolist())
PHONE_SERVICE_CHOICES = sorted(df_raw["PhoneService"].dropna().unique().tolist())
MULTIPLE_LINES_CHOICES = sorted(df_raw["MultipleLines"].dropna().unique().tolist())
INTERNET_SERVICE_CHOICES = sorted(df_raw["InternetService"].dropna().unique().tolist())
ONLINE_SECURITY_CHOICES = sorted(df_raw["OnlineSecurity"].dropna().unique().tolist())
ONLINE_BACKUP_CHOICES = sorted(df_raw["OnlineBackup"].dropna().unique().tolist())
DEVICE_PROTECTION_CHOICES = sorted(df_raw["DeviceProtection"].dropna().unique().tolist())
TECH_SUPPORT_CHOICES = sorted(df_raw["TechSupport"].dropna().unique().tolist())
STREAMING_TV_CHOICES = sorted(df_raw["StreamingTV"].dropna().unique().tolist())
STREAMING_MOVIES_CHOICES = sorted(df_raw["StreamingMovies"].dropna().unique().tolist())
CONTRACT_CHOICES = sorted(df_raw["Contract"].dropna().unique().tolist())
PAPERLESS_BILLING_CHOICES = sorted(df_raw["PaperlessBilling"].dropna().unique().tolist())
PAYMENT_METHOD_CHOICES = sorted(df_raw["PaymentMethod"].dropna().unique().tolist())

# Numerical statistics from CSV
TENURE_MIN = int(df_raw["tenure"].min())
TENURE_MAX = int(df_raw["tenure"].max())
MONTHLY_MIN = float(df_raw["MonthlyCharges"].min())
MONTHLY_MAX = float(df_raw["MonthlyCharges"].max())

# Helper for TotalCharges min/max (treating blank strings gracefully)
tc_clean = pd.to_numeric(df_raw["TotalCharges"], errors="coerce").dropna()
TOTAL_MIN = float(tc_clean.min())
TOTAL_MAX = float(tc_clean.max())


# ==========================================
# 2. PREPROCESSING FUNCTION (EXACT TO TRAINING)
# ==========================================
def preprocess_input(raw_dict: dict) -> pd.DataFrame:
    """
    Preprocess raw input dictionary to match training data transformations:
    1. Wraps raw dict into a single-row DataFrame.
    2. One-hot encodes categorical columns using pd.get_dummies(..., drop_first=True).
    3. Reindexes columns to match model_columns.pkl, filling missing dummies with 0.
    4. Scales numerical columns using scaler.transform on scaler.feature_names_in_.
    """
    df_input = pd.DataFrame([raw_dict])

    # Convert SeniorCitizen from dropdown text or int if needed
    if "SeniorCitizen" in df_input.columns:
        val = df_input["SeniorCitizen"].iloc[0]
        if isinstance(val, str):
            df_input["SeniorCitizen"] = 1 if "1" in val or "Yes" in val else 0
        else:
            df_input["SeniorCitizen"] = int(val)

    # Convert numeric fields
    df_input["tenure"] = pd.to_numeric(df_input["tenure"], errors="coerce").fillna(0)
    df_input["MonthlyCharges"] = pd.to_numeric(df_input["MonthlyCharges"], errors="coerce").fillna(0.0)
    df_input["TotalCharges"] = pd.to_numeric(df_input["TotalCharges"], errors="coerce").fillna(0.0)

    # One-hot encode categorical columns the same way as training
    categorical_cols = df_input.select_dtypes(include=["object"]).columns.tolist()
    df_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

    # Align columns: fill missing dummy columns with 0, drop extras, maintain exact order
    df_input = df_input.reindex(columns=MODEL_COLUMNS, fill_value=0)

    # Scale numeric columns using fitted scaler (only feature_names_in_)
    numerical_cols = SCALER.feature_names_in_.tolist()
    df_input[numerical_cols] = SCALER.transform(df_input[numerical_cols])

    return df_input


# ==========================================
# 3. PREDICTION & SHAP VISUALIZATION
# ==========================================
def predict_churn(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    """
    Handles user submission: validates inputs, preprocesses, predicts probability,
    creates custom styled summary metric cards, and renders a fresh SHAP force plot.
    """
    # 1. Error handling & validation
    try:
        tenure_val = int(tenure)
        if tenure_val < 0:
            raise ValueError("Tenure cannot be negative.")
    except Exception:
        err_card = """
        <div class="result-card error-card">
            <h3>⚠️ Invalid Input</h3>
            <p>Please enter a valid non-negative integer for <b>Tenure</b>.</p>
        </div>
        """
        return err_card, None

    try:
        monthly_val = float(monthly_charges)
        if monthly_val < 0:
            raise ValueError("Monthly Charges cannot be negative.")
    except Exception:
        err_card = """
        <div class="result-card error-card">
            <h3>⚠️ Invalid Input</h3>
            <p>Please enter a valid positive number for <b>Monthly Charges</b>.</p>
        </div>
        """
        return err_card, None

    try:
        total_val = float(total_charges)
        if total_val < 0:
            raise ValueError("Total Charges cannot be negative.")
    except Exception:
        err_card = """
        <div class="result-card error-card">
            <h3>⚠️ Invalid Input</h3>
            <p>Please enter a valid positive number for <b>Total Charges</b> (e.g. 150.50).</p>
        </div>
        """
        return err_card, None

    # Construct raw feature dictionary
    raw_input = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure_val,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_val,
        "TotalCharges": total_val,
    }

    try:
        # Preprocess input
        processed_df = preprocess_input(raw_input)

        # Predict probability
        probabilities = FINAL_MODEL.predict_proba(processed_df)[0]
        churn_prob = probabilities[1]
        stay_prob = probabilities[0]
        prediction = 1 if churn_prob >= 0.5 else 0

        # Determine risk status & custom color styling
        if churn_prob >= 0.60:
            risk_badge = "CRITICAL RISK"
            badge_color = "#ef4444"
            badge_bg = "rgba(239, 68, 68, 0.15)"
            card_border = "#ef4444"
            status_title = "Likely to Churn"
            status_desc = "Customer demonstrates high churn indicators. Immediate retention intervention recommended."
            icon = "🚨"
        elif churn_prob >= 0.35:
            risk_badge = "MODERATE RISK"
            badge_color = "#f59e0b"
            badge_bg = "rgba(245, 158, 11, 0.15)"
            card_border = "#f59e0b"
            status_title = "Elevated Churn Risk"
            status_desc = "Customer shows moderate vulnerability. Consider loyalty discounts or plan reviews."
            icon = "⚠️"
        else:
            risk_badge = "LOW RISK"
            badge_color = "#10b981"
            badge_bg = "rgba(16, 185, 129, 0.15)"
            card_border = "#10b981"
            status_title = "Likely to Stay"
            status_desc = "Customer profile indicates high loyalty and strong account retention."
            icon = "✅"

        pct_churn = churn_prob * 100
        pct_stay = stay_prob * 100

        # Build premium HTML result card
        result_html = f"""
        <div class="result-container" style="border: 1px solid {card_border};">
            <div class="result-header">
                <div class="risk-pill" style="color: {badge_color}; background: {badge_bg}; border: 1px solid {badge_color}44;">
                    <span class="pulse-dot" style="background: {badge_color};"></span>
                    {risk_badge}
                </div>
                <div class="result-icon">{icon}</div>
            </div>
            
            <div class="result-main">
                <h2 class="result-title" style="color: {badge_color};">{status_title}</h2>
                <p class="result-subtitle">{status_desc}</p>
            </div>

            <div class="probability-display">
                <div class="prob-stat">
                    <span class="prob-number" style="color: {badge_color};">{pct_churn:.1f}%</span>
                    <span class="prob-label">Churn Probability</span>
                </div>
                <div class="prob-stat">
                    <span class="prob-number" style="color: #94a3b8;">{pct_stay:.1f}%</span>
                    <span class="prob-label">Retention Probability</span>
                </div>
            </div>

            <div class="progress-bar-container">
                <div class="progress-bar" style="width: {pct_churn}%; background: {badge_color};"></div>
            </div>

            <div class="key-summary-grid">
                <div class="summary-box">
                    <span class="box-label">Contract</span>
                    <span class="box-value">{contract}</span>
                </div>
                <div class="summary-box">
                    <span class="box-label">Internet</span>
                    <span class="box-value">{internet_service}</span>
                </div>
                <div class="summary-box">
                    <span class="box-label">Tenure</span>
                    <span class="box-value">{tenure_val} mos</span>
                </div>
                <div class="summary-box">
                    <span class="box-label">Monthly Bill</span>
                    <span class="box-value">${monthly_val:.2f}</span>
                </div>
            </div>
        </div>
        """

        # Generate fresh SHAP Force Plot per request
        shap_explanation = SHAP_EXPLAINER(processed_df)
        shap_val_single = shap_explanation.values[0]
        base_val = SHAP_EXPLAINER.expected_value

        # Render matplotlib SHAP force plot in memory
        plt.figure(figsize=(15, 3.8), dpi=160)
        shap.plots.force(
            base_val,
            shap_val_single,
            processed_df.iloc[0],
            matplotlib=True,
            show=False,
            text_rotation=12,
        )
        plt.title("SHAP Force Plot — Individual Feature Impact Breakdown", fontsize=11, fontweight="bold", pad=16, color="#0f172a")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=160, facecolor="#ffffff")
        plt.close("all")
        buf.seek(0)
        shap_img = Image.open(buf)

        return result_html, shap_img

    except Exception as e:
        err_card = f"""
        <div class="result-container" style="border: 1px solid #ef4444;">
            <h3 style="color: #ef4444; margin-top:0;">⚠️ Prediction Error</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">An unexpected error occurred during model evaluation:</p>
            <pre style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 8px; color: #fca5a5; font-size: 0.85rem; overflow-x: auto;">{str(e)}</pre>
        </div>
        """
        return err_card, None


# ==========================================
# 4. CUSTOM THEME & CSS STYLING
# ==========================================
custom_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#090d16",
    body_background_fill_dark="#090d16",
    block_background_fill="#111827",
    block_background_fill_dark="#111827",
    block_border_color="#1e293b",
    block_border_color_dark="#1e293b",
    block_title_text_color="#94a3b8",
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#334155",
    input_border_color_dark="#334155",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
)

custom_css = """
/* Global Container Styling */
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding-top: 1.5rem !important;
}

/* Header Styling */
.app-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 10px;
    border-radius: 9999px;
    margin-bottom: 0.75rem;
}
.app-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #f8fafc;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(to right, #ffffff, #cbd5e1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
    max-width: 720px;
    line-height: 1.5;
}

/* Form Group Cards */
.section-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}
.section-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 1rem;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.5rem;
}
.section-title span.icon {
    font-size: 1.2rem;
}

/* Result Card Component */
.result-container {
    background: linear-gradient(145deg, #111827 0%, #0f172a 100%);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
}
.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}
.risk-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 6px 14px;
    border-radius: 9999px;
}
.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,255,255,0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(255,255,255,0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,255,255,0); }
}
.result-icon {
    font-size: 1.8rem;
}
.result-title {
    font-size: 1.65rem;
    font-weight: 800;
    margin: 0 0 0.35rem 0;
}
.result-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0 0 1.25rem 0;
    line-height: 1.45;
}
.probability-display {
    display: flex;
    gap: 2rem;
    padding: 1rem 0;
    border-top: 1px solid #1e293b;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1rem;
}
.prob-stat {
    display: flex;
    flex-direction: column;
}
.prob-number {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.prob-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 4px;
}
.progress-bar-container {
    width: 100%;
    height: 8px;
    background: #1e293b;
    border-radius: 9999px;
    overflow: hidden;
    margin-bottom: 1.25rem;
}
.progress-bar {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.key-summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
}
.summary-box {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 8px 12px;
    display: flex;
    flex-direction: column;
}
.box-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
}
.box-value {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e2e8f0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Submit and Clear Buttons */
.submit-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    border: none !important;
    padding: 12px 24px !important;
}
.submit-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
}

/* SHAP explanation box */
.shap-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 1rem;
    border: 1px solid #e2e8f0;
}
"""


# ==========================================
# 5. GRADIO APPLICATION LAYOUT
# ==========================================
with gr.Blocks(theme=custom_theme, css=custom_css, title="Customer Churn Predictor") as demo:

    # Header section
    gr.HTML(
        """
        <div class="app-header">
            <div class="header-badge">
                <span>⚡</span> Enterprise AI Intelligence
            </div>
            <h1 class="app-title">Customer Churn Intelligence Engine</h1>
            <p class="app-subtitle">
                High-precision predictive analytics powered by tuned XGBoost with real-time SHAP explainability. 
                Configure customer attributes below to generate instant risk classifications and personalized factor attribution.
            </p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # ------------------------------------
        # Left Column: Structured Input Form
        # ------------------------------------
        with gr.Column(scale=5):

            # Section 1: Customer & Demographics
            with gr.Group():
                gr.HTML('<div class="section-title"><span class="icon">👤</span> Demographics & Account Info</div>')
                with gr.Row():
                    gender = gr.Dropdown(
                        label="Gender",
                        choices=GENDER_CHOICES,
                        value="Female",
                        info="Biological gender",
                    )
                    senior_citizen = gr.Dropdown(
                        label="Senior Citizen",
                        choices=SENIOR_CHOICES,
                        value="No (0)",
                        info="Age 65 or older",
                    )
                with gr.Row():
                    partner = gr.Dropdown(
                        label="Partner",
                        choices=PARTNER_CHOICES,
                        value="Yes",
                        info="Has a spouse/partner",
                    )
                    dependents = gr.Dropdown(
                        label="Dependents",
                        choices=DEPENDENTS_CHOICES,
                        value="No",
                        info="Has children/dependents",
                    )
                tenure = gr.Slider(
                    label="Account Tenure (Months)",
                    minimum=TENURE_MIN,
                    maximum=TENURE_MAX,
                    value=5,
                    step=1,
                    info="Months with current subscription (0 to 72)",
                )

            # Section 2: Subscribed Services
            with gr.Group():
                gr.HTML('<div class="section-title"><span class="icon">🌐</span> Telecom & Digital Services</div>')
                with gr.Row():
                    phone_service = gr.Dropdown(
                        label="Phone Service",
                        choices=PHONE_SERVICE_CHOICES,
                        value="Yes",
                    )
                    multiple_lines = gr.Dropdown(
                        label="Multiple Lines",
                        choices=MULTIPLE_LINES_CHOICES,
                        value="No",
                    )
                with gr.Row():
                    internet_service = gr.Dropdown(
                        label="Internet Service",
                        choices=INTERNET_SERVICE_CHOICES,
                        value="Fiber optic",
                    )
                    online_security = gr.Dropdown(
                        label="Online Security",
                        choices=ONLINE_SECURITY_CHOICES,
                        value="No",
                    )
                with gr.Row():
                    online_backup = gr.Dropdown(
                        label="Online Backup",
                        choices=ONLINE_BACKUP_CHOICES,
                        value="No",
                    )
                    device_protection = gr.Dropdown(
                        label="Device Protection",
                        choices=DEVICE_PROTECTION_CHOICES,
                        value="No",
                    )
                with gr.Row():
                    tech_support = gr.Dropdown(
                        label="Tech Support",
                        choices=TECH_SUPPORT_CHOICES,
                        value="No",
                    )
                    streaming_tv = gr.Dropdown(
                        label="Streaming TV",
                        choices=STREAMING_TV_CHOICES,
                        value="No",
                    )
                streaming_movies = gr.Dropdown(
                    label="Streaming Movies",
                    choices=STREAMING_MOVIES_CHOICES,
                    value="No",
                )

            # Section 3: Billing & Contract Details
            with gr.Group():
                gr.HTML('<div class="section-title"><span class="icon">💳</span> Contract & Billing Profile</div>')
                with gr.Row():
                    contract = gr.Dropdown(
                        label="Contract Type",
                        choices=CONTRACT_CHOICES,
                        value="Month-to-month",
                        info="Subscription duration terms",
                    )
                    paperless_billing = gr.Dropdown(
                        label="Paperless Billing",
                        choices=PAPERLESS_BILLING_CHOICES,
                        value="Yes",
                    )
                payment_method = gr.Dropdown(
                    label="Payment Method",
                    choices=PAYMENT_METHOD_CHOICES,
                    value="Electronic check",
                )
                with gr.Row():
                    monthly_charges = gr.Slider(
                        label="Monthly Charges ($)",
                        minimum=round(MONTHLY_MIN, 2),
                        maximum=round(MONTHLY_MAX, 2),
                        value=85.00,
                        step=0.25,
                        info=f"Range: ${MONTHLY_MIN:.2f} – ${MONTHLY_MAX:.2f}",
                    )
                    total_charges = gr.Number(
                        label="Total Charges ($)",
                        value=425.00,
                        minimum=0.0,
                        info="Cumulative billed amount",
                    )

            # Action Buttons
            with gr.Row():
                submit_btn = gr.Button("⚡ Analyze Churn Risk", variant="primary", elem_classes=["submit-btn"], scale=3)
                reset_btn = gr.Button("↺ Reset", variant="secondary", scale=1)

            # Preset Quick-Load Examples
            gr.Examples(
                examples=[
                    [
                        "Female", "No (0)", "Yes", "No", 5, "Yes", "No",
                        "Fiber optic", "No", "No", "No", "No", "No", "No",
                        "Month-to-month", "Yes", "Electronic check", 85.00, 425.00
                    ],
                    [
                        "Male", "No (0)", "Yes", "Yes", 64, "Yes", "Yes",
                        "DSL", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes",
                        "Two year", "No", "Credit card (automatic)", 89.10, 5702.40
                    ],
                    [
                        "Male", "Yes (1)", "No", "No", 12, "Yes", "Yes",
                        "Fiber optic", "No", "Yes", "No", "No", "Yes", "Yes",
                        "Month-to-month", "Yes", "Electronic check", 98.50, 1182.00
                    ],
                    [
                        "Female", "No (0)", "No", "No", 34, "Yes", "No",
                        "DSL", "Yes", "No", "Yes", "No", "No", "No",
                        "One year", "No", "Mailed check", 56.95, 1889.50
                    ],
                ],
                inputs=[
                    gender, senior_citizen, partner, dependents, tenure,
                    phone_service, multiple_lines, internet_service, online_security,
                    online_backup, device_protection, tech_support, streaming_tv,
                    streaming_movies, contract, paperless_billing, payment_method,
                    monthly_charges, total_charges,
                ],
                label="Sample Customer Archetypes (Click to Autofill)",
            )

        # ------------------------------------
        # Right Column: Instant Intelligence & SHAP Plot
        # ------------------------------------
        with gr.Column(scale=5):
            gr.HTML('<div class="section-title"><span class="icon">📊</span> Predictive Risk Assessment</div>')

            # Dynamic Result Output Card
            result_output = gr.HTML(
                value="""
                <div class="result-container" style="border: 1px dashed #334155; text-align: center; padding: 2.5rem 1.5rem;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🔮</div>
                    <h3 style="color: #cbd5e1; font-size: 1.25rem; margin-bottom: 0.5rem;">Awaiting Customer Input</h3>
                    <p style="color: #64748b; font-size: 0.9rem; max-width: 420px; margin: 0 auto;">
                        Click <b>Analyze Churn Risk</b> or select an archetype example on the left to compute live probabilities and feature impact.
                    </p>
                </div>
                """
            )

            # SHAP Force Plot Visualization
            gr.HTML('<div class="section-title" style="margin-top: 1.5rem;"><span class="icon">🔍</span> SHAP Attribution (Explainable AI)</div>')
            shap_output = gr.Image(
                label="SHAP Force Plot Breakdown",
                type="pil",
                interactive=False,
                elem_classes=["shap-box"],
            )

    # Input event bindings
    submit_btn.click(
        fn=predict_churn,
        inputs=[
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
        ],
        outputs=[result_output, shap_output],
    )

    # Reset button handler
    def reset_form():
        default_card = """
        <div class="result-container" style="border: 1px dashed #334155; text-align: center; padding: 2.5rem 1.5rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🔮</div>
            <h3 style="color: #cbd5e1; font-size: 1.25rem; margin-bottom: 0.5rem;">Awaiting Customer Input</h3>
            <p style="color: #64748b; font-size: 0.9rem; max-width: 420px; margin: 0 auto;">
                Click <b>Analyze Churn Risk</b> or select an archetype example on the left to compute live probabilities and feature impact.
            </p>
        </div>
        """
        return (
            "Female", "No (0)", "Yes", "No", 5, "Yes", "No", "Fiber optic",
            "No", "No", "No", "No", "No", "No", "Month-to-month", "Yes",
            "Electronic check", 85.00, 425.00, default_card, None
        )

    reset_btn.click(
        fn=reset_form,
        inputs=[],
        outputs=[
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
            result_output,
            shap_output,
        ],
    )


# ==========================================
# 6. APP ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    print("Launching Customer Churn Predictor Gradio App...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
