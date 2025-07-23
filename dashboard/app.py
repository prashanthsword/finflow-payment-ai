import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# === Base Path Setup ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_payments_dataset.csv')
FAILURE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'failure_predictor.pkl')
FAILURE_ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder.pkl')
FRAUD_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_detector.pkl')
FRAUD_ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'fraud_encoder.pkl')

# === Load Models ===
with open(FAILURE_MODEL_PATH, "rb") as f:
    failure_model = pickle.load(f)

with open(FAILURE_ENCODER_PATH, "rb") as f:
    failure_encoder = pickle.load(f)

with open(FRAUD_MODEL_PATH, "rb") as f:
    fraud_model = pickle.load(f)

with open(FRAUD_ENCODER_PATH, "rb") as f:
    fraud_encoder = pickle.load(f)

# === Streamlit Config ===
st.set_page_config(page_title="FinFlow AI - Payment Risk Dashboard", layout="wide")
st.title("💳 FinFlow AI Dashboard")
st.subheader("Predict transaction failures, detect frauds, and recommend best gateways")

# === Input Sidebar ===
st.sidebar.header("📤 Enter Transaction Details")

gateway = st.sidebar.selectbox("Payment Gateway", ["Razorpay", "Stripe", "Cashfree", "PayU"])
payment_type = st.sidebar.selectbox("Payment Type", ["UPI", "Credit Card", "Debit Card", "Netbanking", "Wallet"])
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
device = st.sidebar.selectbox("Device", ["Android", "iOS", "Desktop"])
amount = st.sidebar.number_input("Amount (₹)", min_value=1.0, value=999.0)
latency_ms = st.sidebar.slider("Latency (ms)", min_value=50, max_value=2000, value=350)
retry_count = st.sidebar.slider("Retry Count", min_value=0, max_value=5, value=0)
hour = st.sidebar.slider("Hour of Day", min_value=0, max_value=23, value=15)

# === Prepare Input DataFrame ===
input_df = pd.DataFrame([{
    "gateway": gateway,
    "payment_type": payment_type,
    "region": region,
    "device": device,
    "amount": amount,
    "latency_ms": latency_ms,
    "retry_count": retry_count,
    "hour": hour
}])

# === Encode & Predict Failure ===
X_cat = failure_encoder.transform(input_df[["gateway", "payment_type", "region", "device"]])
X_num = input_df[["amount", "latency_ms", "retry_count", "hour"]].to_numpy()
X_final = np.hstack((X_cat, X_num))

failure_prob = failure_model.predict_proba(X_final)[0][1]
fail_prediction = failure_model.predict(X_final)[0]

# === Encode & Predict Fraud ===
X_cat_fraud = fraud_encoder.transform(input_df[["gateway", "payment_type", "region", "device"]])
X_final_fraud = np.hstack((X_cat_fraud, X_num))

fraud_prob = fraud_model.predict_proba(X_final_fraud)[0][1]
fraud_prediction = fraud_model.predict(X_final_fraud)[0]

# === Display Results ===
st.markdown("### 🔍 Prediction Results")
st.metric("💥 Failure Probability", f"{failure_prob:.2%}")
st.metric("🚨 Fraud Probability", f"{fraud_prob:.2%}")

if fail_prediction == 1:
    st.success("✅ Transaction is likely to succeed.")
else:
    st.error("❌ High chance of failure.")

if fraud_prediction == 0:
    st.success("🛡️ Transaction seems legit.")
else:
    st.warning("⚠️ Suspicious transaction - possible fraud.")

# === Gateway Recommendation ===
st.markdown("---")
st.markdown("### 🚦 Gateway Recommendation")

# Load full cleaned dataset
df_cleaned = pd.read_csv(DATA_PATH)

gateway_stats = df_cleaned.groupby("gateway").agg({
    "status_encoded": lambda x: 1 - x.mean(),  # failure rate
    "latency_ms": "mean"
}).reset_index()

gateway_stats.columns = ["gateway", "failure_rate", "avg_latency"]
gateway_stats = gateway_stats.sort_values(by=["failure_rate", "avg_latency"])

recommended_gateway = gateway_stats.iloc[0]["gateway"]

st.info(f"📌 Based on current performance, we recommend using: **{recommended_gateway}**")

st.dataframe(gateway_stats.style.format({
    "failure_rate": "{:.2%}",
    "avg_latency": "{:.1f} ms"
}))
