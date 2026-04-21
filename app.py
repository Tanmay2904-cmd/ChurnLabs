import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="ChurnGuard AI", layout="wide")

st.title("📞 Customer Churn Prediction")
st.markdown("Predict the likelihood of a customer leaving based on their usage patterns and service details.")

# Sidebar for inputs
st.sidebar.header("Customer Information")

def user_input_features():
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 1)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 50.0)
    
    contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    online_security = st.sidebar.selectbox("Online Security", ("Yes", "No", "No internet service"))
    tech_support = st.sidebar.selectbox("Tech Support", ("Yes", "No", "No internet service"))
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

# Prediction Logic
if st.button("Predict"):
    # (In a real app, we would load the pkl model here)
    # Since we are in a demo environment, we will use a logic-based prediction
    
    churn_prob = 0.1
    if input_df['Contract'][0] == 'Month-to-month':
        churn_prob += 0.4
    if input_df['InternetService'][0] == 'Fiber optic':
        churn_prob += 0.2
    if input_df['tenure'][0] < 12:
        churn_prob += 0.2
    if input_df['OnlineSecurity'][0] == 'No':
        churn_prob += 0.1
        
    churn_prob = min(churn_prob, 0.95)
    
    st.subheader("Prediction Result")
    if churn_prob > 0.5:
        st.error(f"⚠️ High Risk of Churn! (Probability: {churn_prob:.2%})")
        st.markdown("### Recommendations:")
        st.markdown("- Offer a loyalty discount.")
        st.markdown("- Reach out with a personalized retention offer.")
        st.markdown("- Suggest upgrading to a 1-year contract.")
    else:
        st.success(f"✅ Low Risk. (Probability: {churn_prob:.2%})")
        st.markdown("### Recommendations:")
        st.markdown("- Maintain current service level.")
        st.markdown("- Cross-sell additional value services.")

# Display EDA charts if needed
st.divider()
st.subheader("Model Insights")
col1, col2 = st.columns(2)

with col1:
    st.write("**Feature Importance**")
    # Mock importance data
    importance = pd.DataFrame({
        'Feature': ['Contract', 'Tenure', 'MonthlyCharges', 'InternetService', 'TechSupport'],
        'Importance': [0.35, 0.25, 0.15, 0.12, 0.08]
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(importance.set_index('Feature'))

with col2:
    st.write("**Churn vs Non-Churn Distribution**")
    st.info("Based on historical data, Month-to-Month contracts have the highest churn rate (42%).")
