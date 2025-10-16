import streamlit as st
import pandas as pd
import pickle
import os
import joblib

# ===============================
# 📦 Load Saved Objects
# ===============================
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")

# Load model and preprocessor
model = joblib.load(MODEL_PATH)

# Always load with joblib if you saved with joblib
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ===============================
# 🎨 Streamlit UI Setup
# ===============================
st.set_page_config(page_title="Telco Customer Churn Prediction", layout="centered")

st.title("📞 Telco Customer Churn Prediction App")
st.markdown("Predict whether a telecom customer is likely to **churn (leave the service)** based on demographic and billing data.")

st.sidebar.header("🔧 Enter Customer Details")

# ===============================
# 🧍 User Inputs
# ===============================
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=100)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, step=0.5)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, step=1.0)

# ===============================
# 🧮 Create DataFrame
# ===============================
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges],
})

# ===============================
# 🔍 Prediction
# ===============================
if st.button("🔍 Predict Churn"):
    try:
        processed_input = preprocessor.transform(input_data)
        prediction = model.predict(processed_input)[0]
        prob = model.predict_proba(processed_input)[0][1]

        if prediction == 1:
            st.error(f" This customer is **likely to churn** (Probability: {prob:.2f})")
        else:
            st.success(f" This customer is **likely to stay** (Churn probability: {prob:.2f})")

    except Exception as e:
        st.error(f" Error during prediction: {e}")

st.markdown("---")
st.caption("🧠 Developed by **Sohel Kumar Sahoo** | Telco Customer Churn Prediction Project")
