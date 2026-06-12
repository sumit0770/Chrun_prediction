import streamlit as st
import joblib
import pandas as pd


model = joblib.load("model/model.pkl")
pipeline = joblib.load("model/pipeline.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn risk")


gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", min_value=0, step=1)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.1)


if st.button("Predict Churn Risk"):
    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    df = pd.DataFrame([data])
    X = pipeline.transform(df)
    prob = model.predict_proba(X)[0][1]

    if prob > 0.6:
        risk = " High"
    elif prob > 0.3:
        risk = " Medium"
    else:
        risk = " Low"

    st.subheader(" Prediction Result")
    st.write(f"**Churn Probability:** `{prob:.2f}`")
    st.write(f"**Churn Risk Level:** {risk}")
