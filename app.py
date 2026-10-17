

import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("rain.pkl")

st.set_page_config(page_title="Rain Prediction", layout="centered")
st.title("🌍 Rain Prediction App")

# User Inputs
Temperature = st.number_input("Temperature", min_value=0.0)
Humidity = st.number_input("Humidity", min_value=0.0)
Pressure = st.number_input("Pressure", min_value=0.0)
WindSpeed = st.number_input("WindSpeed", min_value=0.0)
CloudCover = st.number_input("CloudCover", min_value=0.0)
RainToday = st.number_input("RainToday", min_value=0, max_value=1)

# Prediction
if st.button("Predict Tomorrow Rain"):

    input_data = np.array([[
        Temperature,
        Humidity,
        Pressure,
        WindSpeed,
        CloudCover,
        RainToday
    ]])

    # model predicts
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.success(f"Predicted Rain: {prediction}")

    st.info(
        f"Confidence → Rain 0: {probability[0]:.2f}, "
        f"Rain 1: {probability[1]:.2f}"
    )