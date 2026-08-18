import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("Mental_Health_Model.pkl")

st.title("Mental Health Prediction App")

age = st.number_input("Age", 15, 60)
hours = st.number_input("Social Media Hours", 0, 24)

if st.button("Predict"):
    
    data = pd.DataFrame({
        "Age":[age],
        "Social_Media_Hours":[hours]
    })

    prediction = model.predict(data)

    st.success(f"Prediction: {prediction[0]}")