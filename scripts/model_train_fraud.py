<<<<<<< HEAD
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle

# === Load cleaned data ===
df = pd.read_csv("data/cleaned_payments_dataset.csv")

# === Drop rows with no failures — since fraud happens mostly in failed txns ===
df_fraud = df[(df['status'] == 'Failed') | (df['is_fraud'] == 1)]

# === Define features and label ===
X = df_fraud[[
    'gateway', 'payment_type', 'region', 'device',
    'amount', 'latency_ms', 'retry_count', 'hour'
]]
y = df_fraud['is_fraud']

# === One-hot encode categorical variables ===
categorical_cols = ['gateway', 'payment_type', 'region', 'device']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

# Concatenate with numerical features
X_final = pd.concat([
    pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols)),
    X[['amount', 'latency_ms', 'retry_count', 'hour']].reset_index(drop=True)
], axis=1)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# === Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluation ===
y_pred = model.predict(X_test)
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# === Save model and encoder ===
with open("models/fraud_detector.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/fraud_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\n✅ Fraud model and encoder saved to /models/")
=======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle

# === Load cleaned data ===
df = pd.read_csv("data/cleaned_payments_dataset.csv")

# === Drop rows with no failures — since fraud happens mostly in failed txns ===
df_fraud = df[(df['status'] == 'Failed') | (df['is_fraud'] == 1)]

# === Define features and label ===
X = df_fraud[[
    'gateway', 'payment_type', 'region', 'device',
    'amount', 'latency_ms', 'retry_count', 'hour'
]]
y = df_fraud['is_fraud']

# === One-hot encode categorical variables ===
categorical_cols = ['gateway', 'payment_type', 'region', 'device']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

# Concatenate with numerical features
X_final = pd.concat([
    pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols)),
    X[['amount', 'latency_ms', 'retry_count', 'hour']].reset_index(drop=True)
], axis=1)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# === Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluation ===
y_pred = model.predict(X_test)
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# === Save model and encoder ===
with open("models/fraud_detector.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/fraud_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\n✅ Fraud model and encoder saved to /models/")
>>>>>>> 865f9b5df4b8308c40c58246e41a195dda63e8b7
