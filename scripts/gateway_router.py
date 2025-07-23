<<<<<<< HEAD
import pandas as pd
import os

# === Path Setup ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_payments_dataset.csv')

# === STEP 1: Load Data ===
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Cleaned dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# === STEP 2: Gateway Aggregation ===
gateway_stats = df.groupby('gateway').agg({
    'status_encoded': lambda x: 1 - x.mean(),  # failure rate = 1 - success rate
    'latency_ms': 'mean'
}).reset_index()

gateway_stats.columns = ['gateway', 'failure_rate', 'avg_latency']
gateway_stats = gateway_stats.sort_values(by=['failure_rate', 'avg_latency'])

print("✅ Gateway Health Snapshot:\n")
print(gateway_stats)

# === STEP 3: Recommendation Logic ===
def recommend_gateway(threshold_fail=0.15, threshold_latency=500):
    """
    Recommends the best payment gateway based on performance metrics.
    
    Parameters:
        threshold_fail (float): Max allowed failure rate (default: 15%)
        threshold_latency (int): Max allowed average latency in ms (default: 500)
    
    Returns:
        str: Recommended gateway name
    """
    filtered = gateway_stats[
        (gateway_stats['failure_rate'] <= threshold_fail) &
        (gateway_stats['avg_latency'] <= threshold_latency)
    ]

    if not filtered.empty:
        best = filtered.sort_values(by=['failure_rate', 'avg_latency']).iloc[0]
        return best['gateway']
    else:
        return gateway_stats.iloc[0]['gateway']  # fallback to best overall

# === STEP 4: Example Usage ===
if __name__ == "__main__":
    recommended = recommend_gateway()
    print(f"\n✅ Recommended Gateway: {recommended}")
=======
import pandas as pd
import os

# === Path Setup ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_payments_dataset.csv')

# === STEP 1: Load Data ===
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Cleaned dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# === STEP 2: Gateway Aggregation ===
gateway_stats = df.groupby('gateway').agg({
    'status_encoded': lambda x: 1 - x.mean(),  # failure rate = 1 - success rate
    'latency_ms': 'mean'
}).reset_index()

gateway_stats.columns = ['gateway', 'failure_rate', 'avg_latency']
gateway_stats = gateway_stats.sort_values(by=['failure_rate', 'avg_latency'])

print("✅ Gateway Health Snapshot:\n")
print(gateway_stats)

# === STEP 3: Recommendation Logic ===
def recommend_gateway(threshold_fail=0.15, threshold_latency=500):
    """
    Recommends the best payment gateway based on performance metrics.
    
    Parameters:
        threshold_fail (float): Max allowed failure rate (default: 15%)
        threshold_latency (int): Max allowed average latency in ms (default: 500)
    
    Returns:
        str: Recommended gateway name
    """
    filtered = gateway_stats[
        (gateway_stats['failure_rate'] <= threshold_fail) &
        (gateway_stats['avg_latency'] <= threshold_latency)
    ]

    if not filtered.empty:
        best = filtered.sort_values(by=['failure_rate', 'avg_latency']).iloc[0]
        return best['gateway']
    else:
        return gateway_stats.iloc[0]['gateway']  # fallback to best overall

# === STEP 4: Example Usage ===
if __name__ == "__main__":
    recommended = recommend_gateway()
    print(f"\n✅ Recommended Gateway: {recommended}")
>>>>>>> 865f9b5df4b8308c40c58246e41a195dda63e8b7
