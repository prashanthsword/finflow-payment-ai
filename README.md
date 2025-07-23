# 💳 FinFlow AI – Payment Risk Prediction Dashboard

**FinFlow AI** is a complete ML + dashboard project to:
- ✅ Predict payment failures
- 🚨 Detect frauds
- 📌 Recommend the best payment gateway


Built using:
- 🧠 Scikit-learn (ML models)
- 🐼 Pandas & NumPy (Data analysis)
- 📊 Streamlit (Interactive dashboard)
- 🛠️ Python (ETL, backend logic)

---

## 🔧 Project Structure
finflow_payment_ai/
│
├── data/
│ ├── payments_dataset_sample_5k.csv # Raw sample data
│ └── cleaned_payments_dataset.csv # Cleaned after ETL
│
├── models/
│ ├── failure_predictor.pkl # Failure model
│ ├── fraud_detector.pkl # Fraud model
│ ├── encoder.pkl # Encoder for failure model
│ └── fraud_encoder.pkl # Encoder for fraud model
│
├── scripts/
│ ├── etl_pipeline.py # Data cleaning + feature engg
│ ├── model_train_failure.py # Train failure model
│ └── model_train_fraud.py # Train fraud model
│
├── dashboard/
│ └── app.py # Streamlit frontend app
│
├── utils/
│ └── helpers.py # Optional utils (load_pickle etc.)
│
├── requirements.txt # All dependencies
└── README.


---

## ⚙️ Setup Instructions (Run Step by Step)

### ✅ Step 1: Clone and enter project

git clone https://github.com/prashanthsword/finflow_payment_ai.git
cd finflow_payment_ai
python -m venv venv
# Activate:
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

✅ Step 2: Install required libraries
bash
Copy code
pip install -r requirements.txt

✅ Step 3: Run ETL to clean the raw data
bash
Copy code
python scripts/etl_pipeline.py


✅ Step 4: Train both models (failure + fraud)
bash
Copy code
python scripts/model_train_failure.py
python scripts/model_train_fraud.py
✅ Step 5: Launch the Streamlit Dashboard
bash
Copy code
streamlit run dashboard/app.py
📈 Dashboard Features
💥 Predicts failure probability of a transaction

⚠️ Detects fraud risk

📌 Suggests the most reliable gateway (based on failure rate + latency)

🎛️ Input options: Amount, Gateway, Payment Type, Device, Latency, etc.

🧠 Realtime ML prediction using trained models



