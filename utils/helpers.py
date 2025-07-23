import pickle
import numpy as np

def load_pickle(path):
    """
    Loads a pickle file from the given path.
    
    Args:
        path (str): File path to the .pkl file.
    
    Returns:
        object: Unpickled Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_features(df_input, encoder, numerical_cols, categorical_cols):
    """
    Encodes categorical features, combines with numerical features.
    
    Args:
        df_input (DataFrame): Input data
        encoder (OneHotEncoder): Trained encoder
        numerical_cols (list): List of numerical columns
        categorical_cols (list): List of categorical columns
    
    Returns:
        np.array: Final model-ready input
    """
    X_cat = encoder.transform(df_input[categorical_cols])
    X_num = df_input[numerical_cols].to_numpy()
    X_final = np.hstack((X_cat, X_num))
    return X_final


def recommend_gateway(dataframe, fail_thresh=0.15, latency_thresh=500):
    """
    Recommends the best payment gateway based on failure rate and latency.
    
    Args:
        dataframe (DataFrame): Cleaned payment data with 'gateway', 'status_encoded', 'latency_ms'
        fail_thresh (float): Max acceptable failure rate
        latency_thresh (int): Max acceptable latency in ms
    
    Returns:
        str: Name of recommended payment gateway
    """
    gateway_stats = dataframe.groupby("gateway").agg({
        "status_encoded": lambda x: 1 - x.mean(),  # failure rate
        "latency_ms": "mean"
    }).reset_index()

    gateway_stats.columns = ["gateway", "failure_rate", "avg_latency"]

    # Filter gateways under thresholds
    filtered = gateway_stats[
        (gateway_stats["failure_rate"] <= fail_thresh) &
        (gateway_stats["avg_latency"] <= latency_thresh)
    ]

    # Return best match or fallback to overall best
    if not filtered.empty:
        return filtered.sort_values(by=["failure_rate", "avg_latency"]).iloc[0]["gateway"]
    else:
        return gateway_stats.sort_values(by=["failure_rate", "avg_latency"]).iloc[0]["gateway"]
