import pandas as pd
import os

# === STEP 0: Set Up Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'payments_dataset_sample_5k.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_payments_dataset.csv')

# === STEP 1: EXTRACT ===
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Raw dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# === STEP 2: TRANSFORM ===
# Handle missing timestamps
if 'timestamp' not in df.columns:
    raise KeyError("❌ 'timestamp' column missing from dataset.")

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])  # drop rows with invalid timestamps

# Feature Engineering
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()
df['month'] = df['timestamp'].dt.month

# Clean data: remove latency outliers
df = df[df['latency_ms'] < 2000]

# Encode status (Success=1, Failed=0)
df['status_encoded'] = df['status'].map({'Success': 1, 'Failed': 0})

# Optional: reorder columns if all exist
desired_cols = [
    'txn_id', 'user_id', 'gateway', 'payment_type', 'region', 'device',
    'amount', 'status', 'status_encoded', 'is_fraud', 'latency_ms',
    'retry_count', 'timestamp', 'hour', 'day_of_week', 'month'
]
df = df[[col for col in desired_cols if col in df.columns]]

# === STEP 3: LOAD ===
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Cleaned data saved to: {OUTPUT_PATH}")
