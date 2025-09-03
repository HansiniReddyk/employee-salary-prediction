import streamlit as st
import pandas as pd
import joblib

st.title("💼 Employee Salary Prediction App")

# Load trained model
model = joblib.load("salary_model.pkl")

# Input fields
st.header("Enter Employee Details:")

col1, col2 = st.columns(2)

with col1:
    years = st.number_input("Years of Experience", min_value=0, max_value=50, value=3)
    hours = st.number_input("Hours per Week", min_value=1, max_value=80, value=40)
    dept = st.selectbox("Department", ["IT", "HR", "Finance"])

with col2:
    edu = st.selectbox("Education", ["Bachelors", "Masters", "PhD"])
    role = st.text_input("Job Role", "Data Analyst")
    loc = st.selectbox("Location", ["Hyderabad", "Bengaluru", "Pune", "Chennai"])

# Predict button
if st.button("Predict Salary"):
    input_data = pd.DataFrame({
        "YearsExperience": [years],
        "Education": [edu],
        "JobRole": [role],
        "Department": [dept],
        "HoursPerWeek": [hours],
        "Location": [loc]
    })

    salary = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹{salary:,.0f}")