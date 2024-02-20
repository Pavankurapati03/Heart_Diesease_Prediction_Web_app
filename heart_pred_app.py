# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 01:42:17 2024

@author: Pavan
"""

import numpy as np
import pickle
import streamlit as st

heart_disease_model = pickle.load(open('trained_new_model.sav', 'rb'))


def main():
    st.title("Heart Disease Prediction Web App")

    # Sidebar with user input
    st.sidebar.header("User Input")

    # Define a function for input validation
    def validate_input(value):
        try:
            float_value = float(value)
            return float_value
        except ValueError:
            return None

    age = validate_input(st.sidebar.text_input("Age"))
    sex = validate_input(st.sidebar.text_input("Sex (0 = Female, 1 = Male)"))
    cp = validate_input(st.sidebar.text_input("Chest Pain Type (0-3)"))
    trestbps = validate_input(st.sidebar.text_input("Resting Blood Pressure"))
    chol = validate_input(st.sidebar.text_input("Cholesterol"))
    fbs = validate_input(st.sidebar.text_input("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)"))
    restecg = validate_input(st.sidebar.text_input("Resting Electrocardiographic Results (0-2)"))
    thalach = validate_input(st.sidebar.text_input("Maximum Heart Rate Achieved"))
    exang = validate_input(st.sidebar.text_input("Exercise Induced Angina (0 = No, 1 = Yes)"))
    oldpeak = validate_input(st.sidebar.text_input("ST Depression Induced by Exercise Relative to Rest"))
    slope = validate_input(st.sidebar.text_input("Slope of the Peak Exercise ST Segment (0-2)"))
    ca = validate_input(st.sidebar.text_input("Number of Major Vessels Colored by Fluoroscopy (0-3)"))
    thal = validate_input(st.sidebar.text_input("Thalassemia Type (0-3)"))

    # Check if any input is invalid
    if None in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]:
        st.warning("Please enter valid numerical values for all input fields.")
        return

    # Create a numpy array with the user input
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

    # Reshape the input data
    input_data_reshaped = input_data.reshape(1, -1)

    # Make predictions
    prediction = heart_disease_model.predict(input_data_reshaped)

    # Display the prediction
    st.subheader("Prediction:")
    if prediction[0] == 0:
        st.write("The person does not have a heart disease.")
    else:
        st.write("The person has a heart disease.")

if __name__ == "__main__":
    main()