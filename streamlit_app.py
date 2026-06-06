import streamlit as st
import pandas as pd
import joblib

# Load Saved Files

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# Page Configuration

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details and click Predict.")

st.markdown("---")

# User Inputs

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    value=12
)

usage_frequency = st.number_input(
    "Usage Frequency",
    min_value=0,
    value=10
)

support_calls = st.number_input(
    "Support Calls",
    min_value=0,
    value=2
)

payment_delay = st.number_input(
    "Payment Delay",
    min_value=0,
    value=5
)

subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)

contract_length = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Annual"]
)

total_spend = st.number_input(
    "Total Spend",
    min_value=0.0,
    value=500.0
)

last_interaction = st.number_input(
    "Last Interaction",
    min_value=0,
    value=10
)

st.markdown("---")

# Prediction

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Tenure": [tenure],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription_type],
        "Contract Length": [contract_length],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    processed_data = pipeline.transform(input_data)

    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: Customer is likely to CHURN")
    else:
        st.success("✅ Prediction: Customer is likely to STAY")

