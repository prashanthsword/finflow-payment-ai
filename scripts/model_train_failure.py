import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle

# === Load cleaned data ===
df = pd.read_csv("data/cleaned_payments_dataset.csv")

# === Define features and label ===
X = df[[
    'gateway', 'payment_type', 'region', 'device',
    'amount', 'latency_ms', 'retry_count', 'hour'
]]
y = df['status_encoded']  # 1 = Success, 0 = Failed

# === One-hot encode categorical variables ===
categorical_cols = ['gateway', 'payment_type', 'region', 'device']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

# Concatenate encoded + numerical features
X_final = pd.concat([
    pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols)),
    X[['amount', 'latency_ms', 'retry_count', 'hour']].reset_index(drop=True)
], axis=1)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# === Train model ===
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# === Save model + encoder ===
with open("models/failure_predictor.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\n✅ Model and encoder saved to /models/")
