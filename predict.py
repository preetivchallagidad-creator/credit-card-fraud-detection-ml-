import pandas as pd
import joblib
import time

# Load model
model = joblib.load("../models/fraud_model.pkl")

# Load some test data
data = pd.read_csv("../data/creditcard.csv")
X = data.drop("Class", axis=1)

print("Starting Real-Time Fraud Detection Simulation...\n")

for i in range(10):
    transaction = X.iloc[i:i+1]
    prediction = model.predict(transaction)
    prob = model.predict_proba(transaction)[0][1]

    if prediction[0] == 1:
        print(f"Transaction {i} 🚨 FRAUD ALERT! Probability: {prob:.4f}")
    else:
        print(f"Transaction {i} ✅ Legitimate. Probability: {prob:.4f}")

    time.sleep(1)
