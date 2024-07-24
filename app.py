import streamlit as st
import pickle
import numpy as np

# Load the machine learning model
with open('model/random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

st.title("Analisi Kesehatan Mental")

st.write("### Input Data")

snoring_rate = st.number_input("Snoring Rate", min_value=0.0, step=0.1)
respiration_rate = st.number_input("Respiration Rate", min_value=0.0, step=0.1)
body_temperature = st.number_input("Body Temperature (F)", min_value=0.0, step=0.1)
limb_movement = st.number_input("Limb Movement", min_value=0.0, step=0.1)
blood_oxygen = st.number_input("Blood Oxygen", min_value=0.0, step=0.1)
eye_movement = st.number_input("Eye Movement", min_value=0.0, step=0.1)
sleeping_hours = st.number_input("Sleeping Hours", min_value=0.0, step=0.1)
heart_rate = st.number_input("Heart Rate", min_value=0.0, step=0.1)

# Create input array for prediction
input_data = np.array([[snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate]])

if st.button("Predict"):
    # Predict using the loaded model
    prediction = loaded_model.predict(input_data)

    # Mapping integer prediction to string
    stress_level_mapping = {
        0: 'low/normal',
        1: 'medium low',
        2: 'medium',
        3: 'medium high',
        4: 'high'
    }

    # Get the predicted stress level in string format
    prediction_str = stress_level_mapping[prediction[0]]

    st.write(f"### Predicted Stress Level: {prediction_str}")
